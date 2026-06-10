// Copyright © 2026 Apple Inc.
//
// banneker-kv-prefill: turn-free prompt-prefix prefill + in-memory KV snapshot
// reuse. See `ChatSession.prefill(prompt:)` and
// `ChatSession.init(_:resuming:generateParameters:...)`.

import Foundation
import MLX

/// An immutable, reusable snapshot of the KV cache state produced by a
/// turn-free prefill of a prompt prefix (system instructions + user prompt,
/// rendered through the model's chat template, *stopping before* the
/// end-of-user-turn / generation-prompt template tail).
///
/// Build one with ``ChatSession/prefill(prompt:)``, then start any number of
/// generations from it via ``ChatSession/init(_:resuming:generateParameters:processing:additionalContext:tools:toolDispatch:)``.
/// Each generation works on an independent deep copy of the cached KV arrays
/// (``KVCache/copy()``), so the snapshot itself is never mutated and can be
/// reused across many resume calls.
///
/// ## Correctness contract
///
/// A resume **re-tokenizes the full prompt** it was given and verifies, token
/// by token, that the snapshot's consumed tokens are a prefix of the new
/// rendering. On any mismatch that cannot be reconciled by trimming the
/// copied caches, the resume **falls back to a full prefill** — it can be
/// slower than expected, but it can never generate from wrong context.
///
/// Fallback (full fresh prefill) triggers when:
/// - the resuming prompt's token rendering does not extend the snapshot's
///   consumed tokens, and the divergence cannot be repaired by trimming
///   (e.g. a `RotatingKVCache` that has already rotated is not trimmable);
/// - the resuming session's `GenerateParameters.maxKVSize` differs from the
///   one the snapshot was prefilled with (different cache geometry);
/// - `maxKVSize` is set and the snapshot's prefix length is >= `maxKVSize`
///   (the parameter-driven rotating cache would have rotated during prefill,
///   so the early prefix tokens are no longer faithfully represented);
/// - the resuming input carries an attention mask, images, or videos.
///
/// Model-internal sliding-window caches (e.g. Gemma's per-layer
/// `RotatingKVCache(maxSize: slidingWindow)`) are *not* a fallback trigger by
/// themselves: rotation there is the architecture's own sliding-window
/// behavior and a fresh full prefill of the same tokens rotates identically.
/// They only force a fallback in the trim path (a rotated cache cannot be
/// trimmed, so a token-boundary mismatch inside the prefix cannot be
/// repaired).
///
/// ## Memory cost
///
/// Each resume materializes one deep copy of the snapshot's KV arrays. For
/// Gemma-4-E4B (24 cached layers: 4 full-attention `KVCacheSimple` + 20
/// sliding-window `RotatingKVCache(maxSize: 512)`, 2 KV heads × head dim 256,
/// bf16) a ~3K-token prefix costs:
///
/// - per token per layer: 2 (K+V) × 2 heads × 256 dim × 2 bytes = 2 KiB
/// - full-attention layers: 4 × 3,000 × 2 KiB ≈ 24 MiB
/// - sliding-window layers (capped at ~512–1,023 retained entries):
///   20 × ~768 × 2 KiB ≈ 30 MiB
///
/// i.e. **~50–55 MiB held by the snapshot, plus ~50–55 MiB per in-flight
/// resume copy**. Sequential reuse (Banneker's retry/revise/fabrication loop)
/// peaks at ~2× ≈ 110 MiB.
///
/// > Important: `PrefixState` is not thread-safe in the same sense as
/// > `ChatSession`: it may be *consumed* by multiple sequential sessions, but
/// > do not resume from the same instance on two threads at the same time
/// > unless the underlying `MLXArray`s have been evaluated (the prefill path
/// > evaluates them before returning, so the as-built state is safe to share
/// > across sequential tasks).
public final class PrefixState {

    /// The system instructions baked into the snapshot (position 0 of the
    /// rendered prompt). A resuming session re-renders with these.
    public let instructions: String?

    /// The original user prompt whose rendering was prefilled.
    public let prompt: String

    /// The exact token ids consumed into the KV cache. This is a strict
    /// prefix of the chat-template rendering of `instructions` + `prompt`,
    /// cut *before* the template's end-of-user-turn / generation tail so
    /// that any byte-suffix extension of `prompt` still token-extends it.
    let tokens: [Int]

    /// `GenerateParameters.maxKVSize` at prefill time. A resume with a
    /// different value falls back to a full prefill (cache geometry).
    let maxKVSize: Int?

    /// The master KV snapshot. Never handed out directly — every resume
    /// gets `copiedCaches()`.
    private let caches: [KVCache]

    /// Number of prompt tokens whose KV state this snapshot holds.
    public var tokenCount: Int { tokens.count }

    init(
        instructions: String?, prompt: String, tokens: [Int], maxKVSize: Int?,
        caches: [KVCache]
    ) {
        self.instructions = instructions
        self.prompt = prompt
        self.tokens = tokens
        self.maxKVSize = maxKVSize
        self.caches = caches
    }

    /// Independent deep copies of the snapshot caches (safe to mutate).
    func copiedCaches() -> [KVCache] {
        caches.map { $0.copy() }
    }

    /// The result of attempting to resume from a snapshot.
    struct Resumption {
        /// The cache to generate with (either trimmed copies of the snapshot
        /// or a fresh empty cache on fallback).
        let cache: [KVCache]
        /// The input the `TokenIterator` should process (the un-cached
        /// remainder of the prompt, or the full prompt on fallback).
        let input: LMInput
        /// How many prompt tokens were reused from the snapshot
        /// (0 == full-prefill fallback).
        let reusedTokenCount: Int
    }

    /// Try to resume a generation whose *full* tokenized prompt is `input`.
    ///
    /// Verifies the snapshot tokens against the new rendering and returns
    /// either (copied caches + remainder input) or a full-prefill fallback.
    /// Never throws and never produces wrong context: any doubt → fallback.
    func resume(
        input: LMInput, model: any LanguageModel, parameters: GenerateParameters
    ) -> Resumption {
        func fallback() -> Resumption {
            Resumption(
                cache: model.newCache(parameters: parameters),
                input: input,
                reusedTokenCount: 0)
        }

        // Media or masked input — out of scope for prefix reuse.
        guard input.text.mask == nil, input.image == nil, input.video == nil else {
            return fallback()
        }

        // Cache geometry must match the prefill-time parameters.
        guard parameters.maxKVSize == maxKVSize else {
            return fallback()
        }

        // A parameter-driven rotating cache smaller than (or equal to) the
        // prefix would have rotated during prefill — the snapshot no longer
        // faithfully represents the early prefix tokens. Correctness first.
        if let maxKVSize, tokens.count >= maxKVSize {
            return fallback()
        }

        let fullTokens = input.text.tokens.asArray(Int.self)

        // The longest shared token prefix, capped so at least one token
        // remains for the TokenIterator to prime generation with.
        var reuse = commonPrefixLength(tokens, fullTokens)
        reuse = min(reuse, fullTokens.count - 1)
        guard reuse > 0 else { return fallback() }

        let copies = copiedCaches()

        // Boundary mismatch inside the snapshot (e.g. a BPE merge across the
        // prefix/suffix boundary): trim the copies back to the shared prefix.
        // Caches that cannot be trimmed exactly force a fallback.
        let trimNeeded = tokens.count - reuse
        if trimNeeded > 0 {
            for cache in copies {
                // A cache that holds fewer tokens than the divergence point
                // (e.g. an unused KV-shared-layer slot at offset 0) only
                // needs trimming down to its own offset.
                let need = min(cache.offset, trimNeeded)
                if need == 0 { continue }
                guard cache.isTrimmable, cache.trim(need) == need else {
                    return fallback()
                }
            }
        }

        let remainder = MLXArray(fullTokens[reuse...].map { Int32($0) })
        return Resumption(
            cache: copies,
            input: LMInput(text: .init(tokens: remainder)),
            reusedTokenCount: reuse)
    }
}

/// Length of the longest common prefix of two token arrays.
func commonPrefixLength(_ a: [Int], _ b: [Int]) -> Int {
    var i = 0
    let n = min(a.count, b.count)
    while i < n && a[i] == b[i] {
        i += 1
    }
    return i
}
