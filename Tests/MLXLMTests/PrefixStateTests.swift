// Copyright © 2026 Apple Inc.
//
// banneker-kv-prefill: tests for turn-free prompt-prefix prefill + KV
// snapshot reuse (`ChatSession.prefill(prompt:)` / `PrefixState` /
// `ChatSession.init(_:resuming:...)`).

import Foundation
import MLX
import MLXLLM
import MLXNN
import XCTest

@testable import MLXLMCommon

// MARK: - Deterministic fixtures

/// A deterministic, chat-template-shaped tokenizer.
///
/// Rendering: `[90] sys-chars [91] [92] user-chars [93] [94]`
/// where each content character maps 1:1 to a token id in `1...85`
/// (`1 + scalar % 85`). The trailing `[93] [94]` models the real-world
/// end-of-user-turn + generation-prompt template tail: extending the user
/// content token-extends the rendering *up to* that tail, which is exactly
/// the property `prefill` must detect and stop before.
struct PrefixTestTokenizer: MLXLMCommon.Tokenizer {

    static let systemOpen = 90
    static let systemClose = 91
    static let userOpen = 92
    static let userClose = 93
    static let generationTail = 94

    func contentTokens(_ text: String) -> [Int] {
        text.unicodeScalars.map { 1 + Int($0.value) % 85 }
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        contentTokens(text)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { "t\($0)" }.joined(separator: " ")
    }

    func convertTokenToId(_ token: String) -> Int? {
        token.hasPrefix("t") ? Int(token.dropFirst()) : nil
    }

    func convertIdToToken(_ id: Int) -> String? {
        "t\(id)"
    }

    var bosToken: String? = nil
    var eosToken: String? = nil
    /// Outside the model's vocabulary (100) so greedy generation never stops
    /// early — tests bound output with `maxTokens` instead.
    var eosTokenId: Int? { 101 }
    var unknownToken: String? = nil
    var unknownTokenId: Int? { 102 }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        var out: [Int] = []
        for message in messages {
            guard let role = message["role"] as? String,
                let content = message["content"] as? String
            else { continue }
            switch role {
            case "system":
                out.append(Self.systemOpen)
                out.append(contentsOf: contentTokens(content))
                out.append(Self.systemClose)
            default:
                out.append(Self.userOpen)
                out.append(contentsOf: contentTokens(content))
                out.append(Self.userClose)
            }
        }
        out.append(Self.generationTail)
        return out
    }
}

struct PrefixTestProcessor: UserInputProcessor {
    let tokenizer = PrefixTestTokenizer()
    let configuration = ModelConfiguration(id: "prefix-test")
    let messageGenerator = DefaultMessageGenerator()

    func prepare(input: UserInput) throws -> LMInput {
        let messages = messageGenerator.generate(from: input)
        let promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools,
            additionalContext: input.additionalContext)
        return LMInput(tokens: MLXArray(promptTokens))
    }
}

// MARK: - Tests

public class PrefixStateTests: XCTestCase {

    /// Tiny random-weight Gemma 3 text model. `slidingWindow` is
    /// configurable so tests can force the sliding-window `RotatingKVCache`s
    /// to rotate (the production Gemma-4 shape for long prefixes).
    private func makeContainer(slidingWindow: Int = 512) -> ModelContainer {
        let config = Gemma3TextConfiguration(
            modelType: "text",
            hiddenSize: 64, hiddenLayers: 8, intermediateSize: 64, attentionHeads: 4,
            headDim: 64,
            rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 4,
            ropeTheta: 1_000_000, ropeLocalBaseFreq: 10_000,
            ropeTraditional: false, queryPreAttnScalar: 256,
            slidingWindow: slidingWindow, slidingWindowPattern: 6,
            maxPositionEmbeddings: 32768
        )
        let model = Gemma3TextModel(config)
        eval(model)

        let processor = PrefixTestProcessor()
        let context = ModelContext(
            configuration: processor.configuration,
            model: model,
            processor: processor,
            tokenizer: processor.tokenizer)
        return ModelContainer(context: context)
    }

    private let params = GenerateParameters(maxTokens: 16, temperature: 0.0)

    private let system = "You are a contracting assistant."
    private let basePrompt = "Draft the executive summary for solicitation 47QRAA."
    private let repairSuffix = "\n\nREPAIR: replace line 2; keep all citations."

    /// Tokenize a [system?, user] chat through the container's processor and
    /// run `PrefixState.resume` directly, returning `reusedTokenCount`
    /// (0 == full-prefill fallback).
    private func resumeDirect(
        _ state: PrefixState, _ container: ModelContainer, system: String?,
        prompt: String, parameters: GenerateParameters
    ) async throws -> Int {
        var messages: [Chat.Message] = []
        if let system { messages.append(.system(system)) }
        messages.append(.user(prompt))
        let input = try await container.prepare(input: UserInput(chat: messages))

        let inputBox = SendableBox(input)
        let stateBox = SendableBox(state)
        return try await container.perform { context in
            let resumption = stateBox.consume().resume(
                input: inputBox.consume(), model: context.model, parameters: parameters)
            eval(resumption.cache)
            return resumption.reusedTokenCount
        }
    }

    // MARK: Pure logic

    func testCommonPrefixLength() {
        XCTAssertEqual(commonPrefixLength([], []), 0)
        XCTAssertEqual(commonPrefixLength([1, 2, 3], []), 0)
        XCTAssertEqual(commonPrefixLength([1, 2, 3], [1, 2, 3]), 3)
        XCTAssertEqual(commonPrefixLength([1, 2, 3], [1, 2, 3, 4, 5]), 3)
        XCTAssertEqual(commonPrefixLength([1, 2, 3, 9], [1, 2, 3, 4, 5]), 3)
        XCTAssertEqual(commonPrefixLength([9, 2], [1, 2]), 0)
    }

    // MARK: Prefill boundary

    /// The prefill must stop exactly at the end of the user content —
    /// before the `[93] [94]` end-of-turn/generation template tail — so any
    /// byte-suffix extension of the prompt still token-extends the snapshot.
    func testPrefillStopsBeforeTemplateTail() async throws {
        let container = makeContainer()
        let session = ChatSession(container, instructions: system, generateParameters: params)
        let state = try await session.prefill(prompt: basePrompt)

        let tok = PrefixTestTokenizer()
        // [90] sys [91] [92] user — everything before [93] [94].
        let expected =
            1 + tok.contentTokens(system).count + 1 + 1 + tok.contentTokens(basePrompt).count
        XCTAssertEqual(state.tokenCount, expected)

        // And the snapshot tokens are a strict prefix of the full rendering.
        let full = try tok.applyChatTemplate(
            messages: [
                ["role": "system", "content": system],
                ["role": "user", "content": basePrompt],
            ])
        XCTAssertEqual(state.tokens, Array(full[..<expected]))
        XCTAssertLessThan(state.tokenCount, full.count)
    }

    func testPrefillWithoutInstructions() async throws {
        let container = makeContainer()
        let session = ChatSession(container, generateParameters: params)
        let state = try await session.prefill(prompt: basePrompt)
        let tok = PrefixTestTokenizer()
        XCTAssertEqual(state.tokenCount, 1 + tok.contentTokens(basePrompt).count)
        XCTAssertNil(state.instructions)
    }

    // MARK: Resume verification (direct)

    /// Byte-exact prefix superset (the Banneker surgical/repair prompt
    /// shape): the full snapshot is reused, no trimming.
    func testResumeReusesFullPrefixForSupersetPrompt() async throws {
        let container = makeContainer()
        let session = ChatSession(container, instructions: system, generateParameters: params)
        let state = try await session.prefill(prompt: basePrompt)

        let reused = try await resumeDirect(
            state, container, system: system,
            prompt: basePrompt + repairSuffix, parameters: params)
        XCTAssertEqual(reused, state.tokenCount)
    }

    /// Identical prompt (the retry shape): full reuse; the template tail
    /// remains as the un-cached remainder so generation can be primed.
    func testResumeReusesFullPrefixForIdenticalPrompt() async throws {
        let container = makeContainer()
        let session = ChatSession(container, instructions: system, generateParameters: params)
        let state = try await session.prefill(prompt: basePrompt)

        let reused = try await resumeDirect(
            state, container, system: system, prompt: basePrompt, parameters: params)
        XCTAssertEqual(reused, state.tokenCount)
    }

    /// A non-extending prompt diverges inside the snapshot. With trimmable
    /// caches (no rotation) the shared head — template + system prompt — is
    /// still reused and the divergent tail is trimmed off the copies.
    func testResumeTrimsToSharedHeadForDivergentPrompt() async throws {
        let container = makeContainer()
        let session = ChatSession(container, instructions: system, generateParameters: params)
        let state = try await session.prefill(prompt: basePrompt)

        let other = "List three NAICS codes for janitorial services."
        let tok = PrefixTestTokenizer()
        let sharedHead =
            1 + tok.contentTokens(system).count + 1 + 1
            + commonPrefixLength(tok.contentTokens(basePrompt), tok.contentTokens(other))

        let reused = try await resumeDirect(
            state, container, system: system, prompt: other, parameters: params)
        XCTAssertEqual(reused, sharedHead)
        // The master snapshot must be untouched by the trimmed resume.
        XCTAssertEqual(
            state.tokenCount,
            3 + tok.contentTokens(system).count + tok.contentTokens(basePrompt).count)
    }

    /// `maxKVSize` geometry mismatches always fall back to a full prefill.
    func testResumeFallsBackOnMaxKVSizeMismatch() async throws {
        let container = makeContainer()
        let session = ChatSession(container, instructions: system, generateParameters: params)
        let state = try await session.prefill(prompt: basePrompt)

        var rotating = params
        rotating.maxKVSize = 64
        let reused = try await resumeDirect(
            state, container, system: system,
            prompt: basePrompt + repairSuffix, parameters: rotating)
        XCTAssertEqual(reused, 0)
    }

    /// A parameter-driven rotating cache smaller than the prefix falls back
    /// even when prefill and resume parameters agree.
    func testResumeFallsBackWhenPrefixExceedsMaxKVSize() async throws {
        let container = makeContainer()
        var rotating = params
        rotating.maxKVSize = 8

        let session = ChatSession(
            container, instructions: system, generateParameters: rotating)
        let state = try await session.prefill(prompt: basePrompt)
        XCTAssertGreaterThanOrEqual(state.tokenCount, 8)

        let reused = try await resumeDirect(
            state, container, system: system,
            prompt: basePrompt + repairSuffix, parameters: rotating)
        XCTAssertEqual(reused, 0)
    }

    /// Production Gemma shape: sliding-window `RotatingKVCache`s that have
    /// rotated during the prefix. Exact-superset prompts still reuse fully
    /// (no trim needed); divergence inside the prefix cannot be trimmed on a
    /// rotated cache, so it falls back. Correctness first.
    func testRotatedSlidingWindowReuseAndFallback() async throws {
        let container = makeContainer(slidingWindow: 16)
        let session = ChatSession(container, instructions: system, generateParameters: params)
        let state = try await session.prefill(prompt: basePrompt)
        XCTAssertGreaterThan(state.tokenCount, 16, "prefix must overflow the sliding window")

        // Superset → full reuse, no trim required.
        let reusedSuperset = try await resumeDirect(
            state, container, system: system,
            prompt: basePrompt + repairSuffix, parameters: params)
        XCTAssertEqual(reusedSuperset, state.tokenCount)

        // Divergence inside the prefix → trim required → rotated caches are
        // not trimmable → full-prefill fallback.
        var divergent = basePrompt
        divergent.removeLast()
        divergent.append("X")
        let reusedDivergent = try await resumeDirect(
            state, container, system: system,
            prompt: divergent + repairSuffix, parameters: params)
        XCTAssertEqual(reusedDivergent, 0)
    }

    // MARK: End-to-end output identity (greedy)

    /// prefill + resume must produce byte-identical output to a fresh
    /// session at temperature 0 for a prefix-superset prompt.
    func testPrefillResumeMatchesFreshSession() async throws {
        let container = makeContainer()
        let fullPrompt = basePrompt + repairSuffix

        let fresh = ChatSession(container, instructions: system, generateParameters: params)
        let expected = try await fresh.respond(to: fullPrompt)

        let warm = ChatSession(container, instructions: system, generateParameters: params)
        let state = try await warm.prefill(prompt: basePrompt)

        let resumed = ChatSession(container, resuming: state, generateParameters: params)
        let actual = try await resumed.respond(to: fullPrompt)

        XCTAssertEqual(expected, actual)
        XCTAssertFalse(actual.isEmpty)
    }

    /// The same snapshot must be consumable multiple times, each resume on
    /// its own copy, each matching its own fresh baseline.
    func testPrefixStateIsReusableAcrossMultipleResumes() async throws {
        let container = makeContainer()
        let warm = ChatSession(container, instructions: system, generateParameters: params)
        let state = try await warm.prefill(prompt: basePrompt)
        let tokenCountBefore = state.tokenCount

        let suffixes = [repairSuffix, "\n\nREVISE: tighten to 80 words.", repairSuffix, ""]
        for suffix in suffixes {
            let fullPrompt = basePrompt + suffix

            let fresh = ChatSession(
                container, instructions: system, generateParameters: params)
            let expected = try await fresh.respond(to: fullPrompt)

            let resumed = ChatSession(container, resuming: state, generateParameters: params)
            let actual = try await resumed.respond(to: fullPrompt)

            XCTAssertEqual(expected, actual, "suffix: \(suffix.debugDescription)")
        }
        XCTAssertEqual(state.tokenCount, tokenCountBefore)
    }

    /// A resume through the public API with a prompt that does not extend
    /// the prefix must still produce output identical to a fresh session
    /// (head-reuse + trim, or fallback — either way, correct context).
    func testResumeWithDivergentPromptStillCorrect() async throws {
        let container = makeContainer()
        let other = "List three NAICS codes for janitorial services."

        let fresh = ChatSession(container, instructions: system, generateParameters: params)
        let expected = try await fresh.respond(to: other)

        let warm = ChatSession(container, instructions: system, generateParameters: params)
        let state = try await warm.prefill(prompt: basePrompt)

        let resumed = ChatSession(container, resuming: state, generateParameters: params)
        let actual = try await resumed.respond(to: other)

        XCTAssertEqual(expected, actual)
    }

    /// Same identity check under rotated sliding-window caches (the
    /// production Gemma-4 long-prefix shape).
    func testPrefillResumeMatchesFreshWithRotatedSlidingWindow() async throws {
        let container = makeContainer(slidingWindow: 16)
        let fullPrompt = basePrompt + repairSuffix

        let fresh = ChatSession(container, instructions: system, generateParameters: params)
        let expected = try await fresh.respond(to: fullPrompt)

        let warm = ChatSession(container, instructions: system, generateParameters: params)
        let state = try await warm.prefill(prompt: basePrompt)

        let resumed = ChatSession(container, resuming: state, generateParameters: params)
        let actual = try await resumed.respond(to: fullPrompt)

        XCTAssertEqual(expected, actual)
    }

    /// After the prefix is resolved the session behaves like a normal
    /// multi-turn session.
    func testResumedSessionSupportsFollowUpTurns() async throws {
        let container = makeContainer()
        let warm = ChatSession(container, instructions: system, generateParameters: params)
        let state = try await warm.prefill(prompt: basePrompt)

        let resumed = ChatSession(container, resuming: state, generateParameters: params)
        let first = try await resumed.respond(to: basePrompt + repairSuffix)
        XCTAssertFalse(first.isEmpty)
        let second = try await resumed.respond(to: "Shorter.")
        XCTAssertFalse(second.isEmpty)
    }
}
