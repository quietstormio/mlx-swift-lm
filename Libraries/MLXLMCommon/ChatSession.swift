// Copyright © 2025 Apple Inc.

import CoreGraphics
import Foundation
import MLX

/// Simplified API for multi-turn conversations with LLMs and VLMs.
///
/// For example:
///
/// ```swift
/// let modelContainer = try await loadModelContainer(id: "mlx-community/Qwen3-4B-4bit")
/// let session = ChatSession(modelContainer)
/// print(try await session.respond(to: "What are two things to see in San Francisco?"))
/// print(try await session.respond(to: "How about a great place to eat?"))
/// ```
///
/// - Note: `ChatSession` is not thread-safe. Each session should be used from a single
///   task/thread at a time. The underlying `ModelContainer` handles thread safety for
///   model operations.
public final class ChatSession {

    enum Cache {
        case empty
        case kvcache([KVCache])
        case history([Chat.Message])
        /// banneker-kv-prefill: an unresolved prompt-prefix snapshot. Resolved
        /// into `.kvcache` on the first generation, once the full prompt has
        /// been tokenized and verified against the snapshot (see
        /// ``PrefixState/resume(input:model:parameters:)``).
        case prefix(PrefixState)
    }

    private let model: ModelContainer
    public var instructions: String?
    private let cache: SerialAccessContainer<Cache>
    public var processing: UserInput.Processing
    public var generateParameters: GenerateParameters
    public var additionalContext: [String: any Sendable]?
    public var tools: [ToolSpec]?
    public var toolDispatch: (@Sendable (ToolCall) async throws -> String)?

    /// Initialize the `ChatSession`.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContainer``
    ///   - instructions: optional system instructions for the session
    ///   - generateParameters: parameters that control generation
    ///   - processing: media processing configuration for images/videos
    ///   - tools: optional tool specifications
    ///   - toolDispatch: optional tool dispatch -- required for toolcalls if streaming strings rather than details
    ///   - additionalContext: optional model-specific context
    public init(
        _ model: ModelContainer,
        instructions: String? = nil,
        generateParameters: GenerateParameters = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.model = model
        self.instructions = instructions
        self.cache = .init(.empty)
        self.processing = processing
        self.generateParameters = generateParameters
        self.tools = tools
        self.toolDispatch = toolDispatch
        self.additionalContext = additionalContext
    }

    /// Initialize the `ChatSession`.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContext``
    ///   - instructions: optional system instructions for the session
    ///   - generateParameters: parameters that control generation
    ///   - processing: media processing configuration for images/videos
    ///   - tools: optional tool specifications
    ///   - toolDispatch: optional tool dispatch -- required for toolcalls if streaming strings rather than details
    ///   - additionalContext: optional model-specific context
    public init(
        _ model: ModelContext,
        instructions: String? = nil,
        generateParameters: GenerateParameters = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.model = ModelContainer(context: model)
        self.instructions = instructions
        self.cache = .init(.empty)
        self.processing = processing
        self.generateParameters = generateParameters
        self.tools = tools
        self.toolDispatch = toolDispatch
        self.additionalContext = additionalContext
    }

    /// Initialize the `ChatSession` with an existing message history.
    ///
    /// This enables "Prompt Re-hydration" for persistent chat applications.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContainer``
    ///   - instructions: optional system instructions for the session
    ///   - history: The full array of messages to restore (including system prompt)
    ///   - generateParameters: parameters that control generation
    ///   - processing: media processing configuration for images/videos
    ///   - tools: optional tool specifications
    ///   - toolDispatch: optional tool dispatch -- required for toolcalls if streaming strings rather than details
    ///   - additionalContext: optional model-specific context
    public init(
        _ model: ModelContainer,
        instructions: String? = nil,
        history: consuming [Chat.Message],
        generateParameters: GenerateParameters = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.model = model
        self.instructions = instructions
        self.cache = .init(.history(history))
        self.processing = processing
        self.generateParameters = generateParameters
        self.tools = tools
        self.toolDispatch = toolDispatch
        self.additionalContext = additionalContext
    }

    /// Initialize the `ChatSession` with an existing message history.
    ///
    /// This enables "Prompt Re-hydration" for persistent chat applications.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContext``
    ///   - instructions: optional system instructions for the session
    ///   - history: The full array of messages to restore (including system prompt)
    ///   - generateParameters: parameters that control generation
    ///   - processing: media processing configuration for images/videos
    ///   - tools: optional tool specifications
    ///   - toolDispatch: optional tool dispatch -- required for toolcalls if streaming strings rather than details
    ///   - additionalContext: optional model-specific context
    public init(
        _ model: ModelContext,
        instructions: String? = nil,
        history: [Chat.Message],
        generateParameters: GenerateParameters = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.model = ModelContainer(context: model)
        self.instructions = instructions
        self.cache = .init(.history(history))
        self.processing = processing
        self.generateParameters = generateParameters
        self.tools = tools
        self.toolDispatch = toolDispatch
        self.additionalContext = additionalContext
    }

    /// Initialize the `ChatSession` with a pre-built KV cache.
    ///
    /// This enables prefix caching: build a KV cache from a long shared context (e.g. a
    /// system prompt and document) once, save it via ``saveCache(to:)``, and restore it
    /// across multiple sessions to avoid re-prefilling the same tokens each time.
    ///
    /// > Important: If the cache was built from a session that already included system
    /// > instructions, do not pass the same `instructions` here — they would be
    /// > re-tokenized on each call to ``respond(to:role:images:videos:)`` without matching
    /// > KV state, producing incoherent output.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContainer``
    ///   - instructions: optional system instructions for the session — leave `nil` if the
    ///     cache already encodes a system prompt
    ///   - cache: a non-empty `[KVCache]` previously loaded with ``loadPromptCache(url:)``,
    ///     matching the given model
    ///   - generateParameters: parameters that control generation
    ///   - processing: media processing configuration for images/videos
    ///   - tools: optional tool specifications
    ///   - toolDispatch: optional tool dispatch -- required for toolcalls if streaming strings rather than details
    ///   - additionalContext: optional model-specific context
    public init(
        _ model: ModelContainer,
        instructions: String? = nil,
        cache: consuming [KVCache],
        generateParameters: GenerateParameters = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.model = model
        self.instructions = instructions
        self.cache = .init(.kvcache(cache))
        self.processing = processing
        self.generateParameters = generateParameters
        self.tools = tools
        self.toolDispatch = toolDispatch
        self.additionalContext = additionalContext
    }

    /// Initialize the `ChatSession` with a pre-built KV cache.
    ///
    /// This enables prefix caching: build a KV cache from a long shared context (e.g. a
    /// system prompt and document) once, save it via ``saveCache(to:)``, and restore it
    /// across multiple sessions to avoid re-prefilling the same tokens each time.
    ///
    /// > Important: If the cache was built from a session that already included system
    /// > instructions, do not pass the same `instructions` here — they would be
    /// > re-tokenized on each call to ``respond(to:role:images:videos:)`` without matching
    /// > KV state, producing incoherent output.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContext``
    ///   - instructions: optional system instructions for the session — leave `nil` if the
    ///     cache already encodes a system prompt
    ///   - cache: a non-empty `[KVCache]` previously loaded with ``loadPromptCache(url:)``,
    ///     matching the given model
    ///   - generateParameters: parameters that control generation
    ///   - processing: media processing configuration for images/videos
    ///   - tools: optional tool specifications
    ///   - toolDispatch: optional tool dispatch -- required for toolcalls if streaming strings rather than details
    ///   - additionalContext: optional model-specific context
    public init(
        _ model: ModelContext,
        instructions: String? = nil,
        cache: consuming [KVCache],
        generateParameters: GenerateParameters = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.model = ModelContainer(context: model)
        self.instructions = instructions
        self.cache = .init(.kvcache(cache))
        self.processing = processing
        self.generateParameters = generateParameters
        self.tools = tools
        self.toolDispatch = toolDispatch
        self.additionalContext = additionalContext
    }

    /// Initialize the `ChatSession` from a prompt-prefix KV snapshot.
    ///
    /// banneker-kv-prefill: this enables in-memory prefix reuse for prompt
    /// families that share a byte-exact prefix (retry / revise / repair
    /// loops). Build the snapshot once with ``prefill(prompt:)``, then start
    /// each generation from it:
    ///
    /// ```swift
    /// let warm = ChatSession(container, instructions: system, generateParameters: params)
    /// let state = try await warm.prefill(prompt: basePrompt)
    ///
    /// // any number of times, each with its own session:
    /// let session = ChatSession(container, resuming: state, generateParameters: params)
    /// let output = try await session.respond(to: basePrompt + repairSuffix)
    /// ```
    ///
    /// The prompt passed to `respond`/`streamResponse` must be the **full**
    /// prompt (the prefilled prompt plus any suffix). It is re-tokenized and
    /// verified against the snapshot; only the un-cached remainder is
    /// prefilled. On any mismatch the session silently falls back to a full
    /// prefill — output is always correct, reuse is best-effort
    /// (see ``PrefixState`` for the exact fallback conditions).
    ///
    /// The session's instructions are taken from the snapshot (they are part
    /// of the cached prefix). After the first generation the session behaves
    /// like a normal multi-turn session.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContainer`` — must be the same model the
    ///     snapshot was prefilled with
    ///   - prefix: the prompt-prefix snapshot from ``prefill(prompt:)``
    ///   - generateParameters: parameters that control generation. The
    ///     `maxKVSize` must match the prefill-time value or the session
    ///     falls back to a full prefill.
    ///   - processing: media processing configuration for images/videos
    ///   - tools: optional tool specifications
    ///   - toolDispatch: optional tool dispatch
    ///   - additionalContext: optional model-specific context
    public init(
        _ model: ModelContainer,
        resuming prefix: PrefixState,
        generateParameters: GenerateParameters = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.model = model
        self.instructions = prefix.instructions
        self.cache = .init(.prefix(prefix))
        self.processing = processing
        self.generateParameters = generateParameters
        self.tools = tools
        self.toolDispatch = toolDispatch
        self.additionalContext = additionalContext
    }

    /// Run **only the prefill forward pass** for this session's instructions
    /// plus `prompt`, and snapshot the resulting KV cache state.
    ///
    /// banneker-kv-prefill: unlike `respond(to:)` with `maxTokens: 1`, this
    /// performs no decode turn at all — no token is sampled and nothing
    /// beyond the prompt prefix enters the cache, so the snapshot is a clean
    /// prefix that later generations can extend.
    ///
    /// The prompt is rendered through the model's chat template, and the
    /// prefill deliberately stops **before** the template's end-of-user-turn
    /// and generation tail (determined by probe renderings of extended
    /// prompts). This is what makes the snapshot reusable for any
    /// byte-suffix extension of `prompt`: the cached tokens are exactly the
    /// rendering region that is invariant under appending to the user
    /// message.
    ///
    /// This method does not modify the session's own conversation state; the
    /// session can be a throwaway configured with the desired
    /// `instructions` and `generateParameters`.
    ///
    /// - Parameter prompt: the user prompt prefix to prefill
    /// - Returns: a reusable ``PrefixState`` snapshot
    /// - Throws: ``ChatSessionError/prefixUnsupportedInput`` for inputs with
    ///   masks/media or models that carry decoder state;
    ///   ``ChatSessionError/emptyPrefix`` if no stable prefix could be
    ///   determined (e.g. the rendering is empty or one token long)
    public func prefill(prompt: String) async throws -> PrefixState {
        let instructions = self.instructions
        let processing = self.processing
        let tools = self.tools
        let additionalContext = self.additionalContext
        let parameters = self.generateParameters

        let box: SendableBox<PrefixState> = try await model.perform { context in
            func render(_ userContent: String) async throws -> [Int] {
                var messages: [Chat.Message] = []
                if let instructions {
                    messages.append(.system(instructions))
                }
                messages.append(.user(userContent))
                let input = try await context.processor.prepare(
                    input: UserInput(
                        chat: messages, processing: processing,
                        tools: tools, additionalContext: additionalContext))
                guard input.text.mask == nil, input.image == nil, input.video == nil
                else {
                    throw ChatSessionError.prefixUnsupportedInput
                }
                return input.text.tokens.asArray(Int.self)
            }

            let fullTokens = try await render(prompt)

            // Probe renderings: extending the user content perturbs the
            // rendering after the user text but keeps everything before it.
            // The common prefix across the probes is the region invariant
            // under suffix extension — i.e. it excludes the chat template's
            // end-of-user-turn + generation tail and any token that could
            // merge across the boundary. Two dissimilar probes guard against
            // a probe-specific BPE merge at the boundary.
            let probeA = try await render(prompt + "\n\nA")
            let probeB = try await render(prompt + " zq9!")

            var consume = min(
                commonPrefixLength(fullTokens, probeA),
                commonPrefixLength(fullTokens, probeB))
            // Always leave at least one token un-cached so a resume with the
            // identical prompt still has a token to prime generation with.
            consume = min(consume, fullTokens.count - 1)
            guard consume > 0 else {
                throw ChatSessionError.emptyPrefix
            }

            // Turn-free prefill of exactly `consume` tokens: chunked forward
            // passes that only populate the KV cache. No sampling, no decode.
            let cache = context.model.newCache(parameters: parameters)
            let prefillInput = LMInput(text: .init(tokens: MLXArray(fullTokens[..<consume].map { Int32($0) })))
            var decoderState: LMOutput.State? = nil
            switch try context.model.prepare(
                prefillInput, cache: cache, windowSize: parameters.prefillStepSize)
            {
            case .tokens(let remainder):
                if remainder.tokens.size > 0 {
                    let result = context.model(
                        remainder[text: .newAxis], cache: cache, state: nil)
                    decoderState = result.state
                }
            case .logits(let output):
                // The model consumed the whole prefix while preparing; the
                // computed logits are simply discarded (no sampling).
                decoderState = output.state
            }
            // Models that carry decoder-side state (e.g. cross-attention)
            // cannot be snapshotted by KV cache alone.
            guard decoderState == nil else {
                throw ChatSessionError.prefixUnsupportedInput
            }
            // Materialize the cache contents before the snapshot crosses
            // task boundaries (MLXArray contract).
            eval(cache)

            return SendableBox(
                PrefixState(
                    instructions: instructions,
                    prompt: prompt,
                    tokens: Array(fullTokens[..<consume]),
                    maxKVSize: parameters.maxKVSize,
                    caches: cache))
        }
        return box.consume()
    }

    /// Produces a response to a prompt.
    ///
    /// - Parameters:
    ///   - prompt: the user prompt
    ///   - role: the message role (defaults to `.user`)
    ///   - images: list of images (for use with VLMs)
    ///   - videos: list of videos (for use with VLMs)
    /// - Returns: the model's response
    public func respond(
        to prompt: String,
        role: Chat.Message.Role = .user,
        images: consuming [UserInput.Image],
        videos: consuming [UserInput.Video]
    ) async throws -> String {
        var output = ""
        for try await chunk in streamResponse(
            to: prompt, role: role, images: images, videos: videos
        ) {
            output += chunk
        }
        return output
    }

    /// Produces a response to a prompt.
    ///
    /// - Parameters:
    ///   - prompt: the user prompt
    ///   - role: the message role (defaults to `.user`)
    ///   - image: optional image (for use with VLMs)
    ///   - video: optional video (for use with VLMs)
    /// - Returns: the model's response
    public func respond(
        to prompt: String,
        role: Chat.Message.Role = .user,
        image: UserInput.Image? = nil,
        video: UserInput.Video? = nil
    ) async throws -> String {
        try await respond(
            to: prompt,
            role: role,
            images: image.map { [$0] } ?? [],
            videos: video.map { [$0] } ?? []
        )
    }

    /// Produces a streaming response to a prompt as Strings.
    ///
    /// - Parameters:
    ///   - prompt: the user prompt
    ///   - role: the message role (defaults to `.user`)
    ///   - images: list of images (for use with VLMs)
    ///   - videos: list of videos (for use with VLMs)
    /// - Returns: a stream of string chunks from the model
    public func streamResponse(
        to prompt: String,
        role: Chat.Message.Role = .user,
        images: consuming [UserInput.Image],
        videos: consuming [UserInput.Video]
    ) -> AsyncThrowingStream<String, Error> {
        streamMap(to: prompt, role: role, images: images, videos: videos) {
            $0.chunk
        }
    }

    /// Produces a streaming response to a prompt as `Generation`.
    ///
    /// - Parameters:
    ///   - prompt: the user prompt
    ///   - role: the message role (defaults to `.user`)
    ///   - images: list of images (for use with VLMs)
    ///   - videos: list of videos (for use with VLMs)
    /// - Returns: a stream of `Generation` from the model
    public func streamDetails(
        to prompt: String,
        role: Chat.Message.Role = .user,
        images: consuming [UserInput.Image],
        videos: consuming [UserInput.Video]
    ) -> AsyncThrowingStream<Generation, Error> {
        streamMap(to: prompt, role: role, images: images, videos: videos) {
            $0
        }
    }

    /// Produces a streaming response to a prompt by transforming the
    /// raw `Generation` values.
    ///
    /// - Parameters:
    ///   - prompt: the user prompt
    ///   - images: list of images (for use with VLMs)
    ///   - videos: list of videos (for use with VLMs)
    /// - Returns: a stream of transformed values from the model
    private func streamMap<R: Sendable>(
        to prompt: String,
        role: Chat.Message.Role,
        images: consuming [UserInput.Image],
        videos: consuming [UserInput.Video],
        transform: @Sendable @escaping (Generation) -> R?
    ) -> AsyncThrowingStream<R, Error> {
        let (stream, continuation) = AsyncThrowingStream<R, Error>.makeStream()

        // images and videos are not Sendable (MLXArray) but they are consumed
        // and are only being sent to the inner async
        let message = SendableBox<Chat.Message>(
            .init(role: role, content: prompt, images: images, videos: videos)
        )

        let task = Task {
            [
                model,
                instructions, processing, tools, toolDispatch,
                additionalContext, cache, generateParameters
            ] in
            do {
                try await cache.update { cache in

                    // these are all Sendable
                    let processor = await model.processor
                    let tokenizer = await model.tokenizer
                    let modelConfiguration = await model.configuration

                    var messages: [Chat.Message] = []
                    if let instructions {
                        messages.append(.system(instructions))
                    }

                    // prepare the cache, if needed.  note:
                    // this is using the LanguageModel (not Sendable) outside
                    // the protective lock.  Assuming the weights are not
                    // being mutated behind the scenes, this will obey the MLXArray
                    // contract that they be evaluated if used across threads.
                    // This is internal to the implementation and this technique
                    // should not be used in calling code.
                    //
                    // The benefit is that callers can be running multiple
                    // ChatSessions in parallel, as long as the instances
                    // are distinct.  In particular the KVCache cannot
                    // be shared and that is the lock that is held here.

                    let model = await model.perform { context in
                        SendableBox(context.model)
                    }.consume()

                    var kvCache: [KVCache]
                    // banneker-kv-prefill: a prefix snapshot can only be
                    // resolved once the full prompt has been tokenized, so
                    // resolution happens inside the loop below on the first
                    // pass.
                    var pendingPrefix: PrefixState? = nil
                    switch cache {
                    case .empty:
                        kvCache = model.newCache(parameters: generateParameters)
                        cache = .kvcache(kvCache)

                    case .kvcache(let array):
                        kvCache = array

                    case .history(let history):
                        // the KVCache is represented by a chat history
                        kvCache = model.newCache(parameters: generateParameters)
                        cache = .kvcache(kvCache)
                        messages.append(contentsOf: history)

                    case .prefix(let prefixState):
                        pendingPrefix = prefixState
                        kvCache = []  // resolved below
                    }

                    // prepare the input
                    messages.append(message.consume())

                    // loop can restart on tool calls
                    restart: while !messages.isEmpty {
                        let userInput = UserInput(
                            chat: messages, processing: processing,
                            tools: tools, additionalContext: additionalContext)
                        var input = try await processor.prepare(input: userInput)
                        messages.removeAll()

                        // banneker-kv-prefill: verify the tokenized prompt
                        // against the prefix snapshot and either reuse copied
                        // KV state (processing only the remainder) or fall
                        // back to a full prefill. Never wrong context.
                        if let prefixState = pendingPrefix {
                            pendingPrefix = nil
                            let resumption = prefixState.resume(
                                input: input, model: model,
                                parameters: generateParameters)
                            kvCache = resumption.cache
                            input = resumption.input
                            cache = .kvcache(kvCache)
                        }

                        // generate output
                        let iterator = try TokenIterator(
                            input: input, model: model, cache: kvCache,
                            parameters: generateParameters)

                        let (stream, task) = MLXLMCommon.generateTask(
                            promptTokenCount: input.text.tokens.size,
                            modelConfiguration: modelConfiguration,
                            tokenizer: tokenizer,
                            iterator: iterator
                        )

                        var pendingToolCalls: [ToolCall] = []

                        for await item in stream {
                            // collect tool calls for dispatch; if no
                            // toolDispatch the caller handles them via
                            // the transform (streamDetails path)
                            if let toolCall = item.toolCall, toolDispatch != nil {
                                pendingToolCalls.append(toolCall)
                            } else if let value = transform(item) {
                                if case .terminated = continuation.yield(value) {
                                    break
                                }
                            }
                        }

                        // wait for the task to complete -- this is important in
                        // the case where we broke the loop early as the generation
                        // work may continue (briefly) and use the KVCache
                        await task.value

                        // dispatch all tool calls from this generation pass
                        if let toolDispatch, !pendingToolCalls.isEmpty,
                            !Task.isCancelled
                        {
                            for toolCall in pendingToolCalls {
                                let toolResult = try await toolDispatch(toolCall)
                                messages.append(.tool(toolResult))
                            }
                            continue restart
                        }
                    }

                    continuation.finish()
                }
            } catch {
                continuation.finish(throwing: error)
            }
        }

        continuation.onTermination = { _ in
            task.cancel()
        }

        return stream
    }

    /// Produces a streaming response to a prompt.
    ///
    /// - Parameters:
    ///   - prompt: the user prompt
    ///   - image: optional image (for use with VLMs)
    ///   - video: optional video (for use with VLMs)
    /// - Returns: a stream of string chunks from the model
    public func streamResponse(
        to prompt: String,
        image: UserInput.Image? = nil,
        video: UserInput.Video? = nil
    ) -> AsyncThrowingStream<String, Error> {
        streamResponse(
            to: prompt,
            images: image.map { [$0] } ?? [],
            videos: video.map { [$0] } ?? []
        )
    }

    /// Clear the session history and cache, preserving system instructions.
    public func clear() async {
        await cache.update { cache in
            cache = .empty
        }
    }

    /// Wait for exclusive access to the KVCache.
    ///
    /// This is useful for cases where a program is terminating and wants to ensure that any
    /// async operations are complete.
    public func synchronize() async {
        await cache.read { _ in }
    }

    /// Visit the current cache value, if realized as a `[KVCache]`.
    ///
    /// This method is meant for test support.
    func withCache<R: Sendable>(_ body: @Sendable ([KVCache]?) async throws -> R) async rethrows
        -> R?
    {
        try await cache.read { cache in
            switch cache {
            case .kvcache(let cache):
                return try await body(cache)
            default:
                return try await body(nil)
            }
        }
    }

    /// Saves the current KV cache to disk.
    ///
    /// Use one of the initializers that accept a `cache` parameter together with
    /// ``loadPromptCache(url:)`` to restore the saved cache in a future session.
    ///
    /// - Parameter url: the file URL to write the cache to
    /// - Throws: ``ChatSessionError/noCacheAvailable`` if no generation has occurred yet,
    ///   or any error thrown by the underlying file write
    public func saveCache(to url: URL) async throws {
        try await cache.read { cache in
            switch cache {
            case .kvcache(let cache):
                try savePromptCache(url: url, cache: cache)
            default:
                throw ChatSessionError.noCacheAvailable
            }
        }
    }
}

/// Errors thrown by ``ChatSession``.
public enum ChatSessionError: LocalizedError {
    /// ``ChatSession/saveCache(to:)`` was called before any generation occurred.
    case noCacheAvailable

    /// ``ChatSession/prefill(prompt:)`` was called with input that cannot be
    /// snapshotted (attention mask, images/videos, or a model that carries
    /// decoder-side state outside the KV cache).
    case prefixUnsupportedInput

    /// ``ChatSession/prefill(prompt:)`` could not determine a non-empty
    /// stable prompt prefix to cache.
    case emptyPrefix

    public var errorDescription: String? {
        switch self {
        case .noCacheAvailable:
            return
                "No KV cache is available. Call respond() or streamResponse() before saveCache(to:)."
        case .prefixUnsupportedInput:
            return
                "prefill(prompt:) supports text-only prompts on models whose generation state lives entirely in the KV cache."
        case .emptyPrefix:
            return "prefill(prompt:) could not determine a non-empty stable prompt prefix."
        }
    }
}
