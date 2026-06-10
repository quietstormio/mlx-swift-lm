// Copyright © 2026 Apple Inc.
//
// banneker-kv-prefill: env-gated REAL-MODEL harness for turn-free prefill +
// KV snapshot reuse. Skipped unless `PREFIX_E2E_MODEL_DIR` points at a local
// MLX model directory (config.json + weights + tokenizer files), e.g. a
// Gemma-4-E4B checkpoint:
//
// ```sh
// PREFIX_E2E_MODEL_DIR=~/path/to/gemma-4-e4b-it-4bit \
//   xcrun xctest -XCTest PrefixStateRealModelTests \
//   .build/dd/Build/Products/Debug/MLXLMTests.xctest
// ```
//
// (Build first with `xcodebuild build-for-testing -scheme
// mlx-swift-lm-Package -destination 'platform=macOS' -derivedDataPath
// .build/dd -skipMacroValidation`. Plain `swift test` cannot load the Metal
// library in this repo — pre-existing for all GPU tests.)
//
// Hard assertions (deterministic on real weights):
// 1. prefill produces a non-empty snapshot that stops before the chat
//    template tail;
// 2. resume verification reuses the FULL snapshot for both the
//    prefix-superset (repair) and identical-prompt (retry) shapes;
// 3. resumed generations are non-empty, and the per-call prompt-processing
//    saving is reported (prefill reuse vs fresh full prefill).
//
// Strict token-identity (`PREFIX_E2E_STRICT=1`) additionally requires
// resumed output == fresh output, byte for byte. CAVEAT: the resumed path
// prefills with a chunk boundary at the prefix end while the fresh path
// processes the whole prompt at once; quantized matmul kernels differ by
// float-reassociation noise across chunk shapes, so greedy argmax can flip
// at a genuine near-tie and outputs legitimately diverge from that token
// on. Observed on Gemma-4-E4B-4bit (2026-06-10): the superset arm was
// bitwise-identical across 64 greedy tokens; the identical-prompt arm
// diverged at ~token 8 on a near-tie ("support" vs "execute"), both
// outputs coherent and correctly grounded. The fp32 tiny-model suite in
// PrefixStateTests proves the machinery is token-exact when numerics
// cooperate; strict mode here documents real-quantized-kernel behavior.

import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import Tokenizers
import XCTest

@testable import MLXLMCommon

public class PrefixStateRealModelTests: XCTestCase {

    private func reusedTokenCount(
        _ state: PrefixState, _ container: ModelContainer, system: String,
        prompt: String, parameters: GenerateParameters
    ) async throws -> Int {
        let input = try await container.prepare(
            input: UserInput(chat: [.system(system), .user(prompt)]))
        let inputBox = SendableBox(input)
        let stateBox = SendableBox(state)
        return try await container.perform { context in
            let resumption = stateBox.consume().resume(
                input: inputBox.consume(), model: context.model, parameters: parameters)
            eval(resumption.cache)
            return resumption.reusedTokenCount
        }
    }

    func testRealModelPrefillResume() async throws {
        guard let dir = ProcessInfo.processInfo.environment["PREFIX_E2E_MODEL_DIR"] else {
            throw XCTSkip(
                "Set PREFIX_E2E_MODEL_DIR to a local MLX model directory to run this harness")
        }
        let strict = ProcessInfo.processInfo.environment["PREFIX_E2E_STRICT"] == "1"

        let url = URL(filePath: (dir as NSString).expandingTildeInPath)
        let container = try await LLMModelFactory.shared.loadContainer(
            from: url, using: #huggingFaceTokenizerLoader())

        let params = GenerateParameters(maxTokens: 64, temperature: 0.0)
        let system = "You are a federal proposal drafting assistant. Cite facts as [[F1]]."
        let facts = (1 ... 12).map {
            "[[F\($0)]] ACME Corp fact number \($0): completed task order \($0) "
                + "for the Air Force Research Laboratory on schedule and within budget."
        }.joined(separator: "\n")
        let basePrompt = """
            FACTS:
            [[F1]] ACME Corp holds ISO 9001.
            [[F2]] ACME completed the 2024 AFRL data migration ($2.6M).
            \(facts)
            TASK: Draft a 60-word executive summary for solicitation 47QRAA-25-R-0001.
            """
        let repairSuffix = "\n\nREPAIR: the draft cited [[F99]] which does not exist. Remove it."

        // Snapshot the shared prefix once.
        let warm = ChatSession(container, instructions: system, generateParameters: params)
        let prefillStart = Date()
        let state = try await warm.prefill(prompt: basePrompt)
        let prefillTime = Date().timeIntervalSince(prefillStart)
        XCTAssertGreaterThan(state.tokenCount, 0)
        print("[prefix-harness] prefix tokens=\(state.tokenCount) prefill=\(prefillTime)s")

        // Deterministic context verification: both prompt shapes must reuse
        // the FULL snapshot (no trim, no fallback).
        let reusedSuperset = try await reusedTokenCount(
            state, container, system: system,
            prompt: basePrompt + repairSuffix, parameters: params)
        XCTAssertEqual(reusedSuperset, state.tokenCount, "superset prompt must reuse full prefix")
        let reusedIdentical = try await reusedTokenCount(
            state, container, system: system, prompt: basePrompt, parameters: params)
        XCTAssertEqual(
            reusedIdentical, state.tokenCount, "identical prompt must reuse full prefix")

        // Generation arms: fresh vs resumed, same snapshot consumed twice.
        for (label, prompt) in [
            ("superset", basePrompt + repairSuffix), ("identical", basePrompt),
        ] {
            let fresh = ChatSession(container, instructions: system, generateParameters: params)
            let freshStart = Date()
            let expected = try await fresh.respond(to: prompt)
            let freshTime = Date().timeIntervalSince(freshStart)

            let resumed = ChatSession(container, resuming: state, generateParameters: params)
            let resumedStart = Date()
            let actual = try await resumed.respond(to: prompt)
            let resumedTime = Date().timeIntervalSince(resumedStart)

            XCTAssertFalse(actual.isEmpty, "\(label): resumed generation must produce output")
            let common = zip(expected, actual).prefix(while: { $0 == $1 }).count
            print(
                "[prefix-harness] \(label): fresh=\(freshTime)s resumed=\(resumedTime)s "
                    + "identical=\(expected == actual) commonPrefixChars=\(common)/\(expected.count)"
            )
            if strict {
                XCTAssertEqual(
                    expected, actual,
                    "\(label): strict token-identity failed (see near-tie caveat in header)")
            }
        }
    }
}
