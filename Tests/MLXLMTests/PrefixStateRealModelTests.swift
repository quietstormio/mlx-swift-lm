// Copyright © 2026 Apple Inc.
//
// banneker-kv-prefill: env-gated REAL-MODEL harness for turn-free prefill +
// KV snapshot reuse. Skipped unless `PREFIX_E2E_MODEL_DIR` points at a local
// MLX model directory (config.json + weights + tokenizer files), e.g. a
// Gemma-4-E4B fused checkpoint:
//
// ```sh
// PREFIX_E2E_MODEL_DIR=~/path/to/gemma-4-e4b-it-4bit \
//   swift test --filter PrefixStateRealModelTests
// ```
//
// What it asserts, at temperature 0:
// 1. prefill(prefix) + resume(prefix + suffix)  == fresh(prefix + suffix)
// 2. prefill(prefix) + resume(prefix)           == fresh(prefix)   (retry shape)
// 3. the same PrefixState serves both resumes (multi-consume)
//
// Note on bitwise determinism: the resumed path prefills the prompt with a
// chunk boundary at the prefix end, while the fresh path uses the default
// 512-token chunking. Matmul tiling can differ, so logits can differ by
// float-reassociation noise (~1e-6 relative). Greedy argmax is stable to
// this in practice; if a real model ever produces a near-tie at one
// position, outputs may legitimately diverge from that token on. The tiny-
// model tests in PrefixStateTests cover the machinery deterministically;
// this harness is the real-weights confirmation.

import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers
import XCTest

public class PrefixStateRealModelTests: XCTestCase {

    func testRealModelPrefillResumeTokenIdentical() async throws {
        guard let dir = ProcessInfo.processInfo.environment["PREFIX_E2E_MODEL_DIR"] else {
            throw XCTSkip(
                "Set PREFIX_E2E_MODEL_DIR to a local MLX model directory to run this harness")
        }

        let url = URL(filePath: (dir as NSString).expandingTildeInPath)
        let container = try await LLMModelFactory.shared.loadContainer(
            from: url, using: #huggingFaceTokenizerLoader())

        let params = GenerateParameters(maxTokens: 64, temperature: 0.0)
        let system = "You are a federal proposal drafting assistant. Cite facts as [[F1]]."
        let basePrompt = """
            FACTS: [[F1]] ACME Corp holds ISO 9001. [[F2]] ACME completed the \
            2024 AFRL data migration ($2.6M).
            TASK: Draft a 60-word executive summary for solicitation 47QRAA-25-R-0001.
            """
        let repairSuffix = "\n\nREPAIR: the draft cited [[F3]] which does not exist. Remove it."

        // Snapshot the shared prefix once.
        let warm = ChatSession(container, instructions: system, generateParameters: params)
        let state = try await warm.prefill(prompt: basePrompt)
        XCTAssertGreaterThan(state.tokenCount, 0)

        // 1. Superset prompt (surgical/repair shape).
        let fullPrompt = basePrompt + repairSuffix
        let fresh1 = ChatSession(container, instructions: system, generateParameters: params)
        let expected1 = try await fresh1.respond(to: fullPrompt)
        let resumed1 = ChatSession(container, resuming: state, generateParameters: params)
        let actual1 = try await resumed1.respond(to: fullPrompt)
        XCTAssertEqual(expected1, actual1, "superset resume diverged from fresh session")

        // 2. Identical prompt (retry shape) — same snapshot, second consume.
        let fresh2 = ChatSession(container, instructions: system, generateParameters: params)
        let expected2 = try await fresh2.respond(to: basePrompt)
        let resumed2 = ChatSession(container, resuming: state, generateParameters: params)
        let actual2 = try await resumed2.respond(to: basePrompt)
        XCTAssertEqual(expected2, actual2, "identical-prompt resume diverged from fresh session")
    }
}
