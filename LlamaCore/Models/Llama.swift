//
//  Llama.swift
//  llamaTest
//
//  Created by Bruce Burgess on 8/21/25.
//

import Foundation

@MainActor
public class Llama {
    private let llama: LlamaState
    
    /// Creates a new Llama instance.
    /// - Note: You must call `initializeModel(at:)` before generating responses.
    public init() {
        self.llama = LlamaState()
    }
    
    /// Delegate that receives events, tokens, and benchmark messages from the model.
    public weak var delegate: LlamaDelegate? {
        get { llama.delegate }
        set { llama.delegate = newValue }
    }
    
    /// Indicates whether a model is currently loaded in memory.
    /// - Returns: `true` if a model is loaded, otherwise `false`.
    public func isModelLoaded() -> Bool {
        llama.isModelLoaded
    }
    
    /// Indicates whether a response is currently being generated.
    /// - Returns: `true` if the model is in the middle of response generation, otherwise `false`.
    public func isGeneratingResponse() -> Bool {
        llama.isGeneratingResponse
    }
    
    /// Sets the tokens that will cause generation to stop if encountered.
    /// - Parameter tokens: An array of stop sequences.
    public func setStopTokens(tokens: [String]) {
        llama.setStopTokens(tokens: tokens)
    }
    
    /// Sets the maximum number of tokens that the model is allowed to generate.
    /// - Parameter maxToken: The maximum number of tokens to generate.
    public func setMaxToken(maxToken: Int) {
        llama.setMaxToken(maxToken: maxToken)
    }
    
    /// Clears the current model context and state (tokens, KV cache, etc.).
    public func clear() async {
        await llama.clear()
    }
    
    /// Unloads the model from memory, freeing associated resources.
    /// - Note: After calling this, you must reload a model using `initializeModel(at:)` before generating responses.
    public func unloadModel() {
        llama.unloadModel()
    }
    
    /// Loads a model from the specified path with configurable parameters.
    /// - Parameters:
    ///   - path: File path to the model file.
    ///   - temperature: Sampling temperature (controls randomness).
    ///   - distribution: Random seed for sampling.
    ///   - batchCapacity: Number of tokens processed per batch.
    ///   - maxSequenceIdsPerToken: Maximum sequence IDs per token.
    ///   - embeddingSize: Embedding vector size (if used).
    ///   - log: If `true`, appends status messages to `messageLog`.
    ///   - completion: A closure called with `.success` if the model loads, or `.failure` if loading fails.
    public func initializeModel(at path: String, temperature: Float = 0.5, distribution: UInt32 = 1234, batchCapacity: Int32 = 512, maxSequenceIdsPerToken: Int32 = 1, embeddingSize: Int32 = 0, log: Bool = false, completion: @escaping (Result<Void, Error>) -> Void) {
        llama.loadModel(at: path, temperature: temperature, distribution: distribution, batchCapacity: batchCapacity, maxSequenceIdsPerToken: maxSequenceIdsPerToken, embeddingSize: embeddingSize, log: log, completion: completion)
    }
    
    /// Loads a model from the specified path asynchronously.
    /// - Parameters:
    ///   - path: File path to the model file.
    ///   - temperature: Sampling temperature (controls randomness).
    ///   - distribution: Random seed for sampling.
    ///   - batchCapacity: Number of tokens processed per batch.
    ///   - maxSequenceIdsPerToken: Maximum sequence IDs per token.
    ///   - embeddingSize: Embedding vector size (if used).
    ///   - log: If `true`, appends status messages to `messageLog`.
    /// - Throws: An error if the model cannot be loaded.
    public func initializeModel(at path: String, temperature: Float = 0.5, distribution: UInt32 = 1234, batchCapacity: Int32 = 512, maxSequenceIdsPerToken: Int32 = 1, embeddingSize: Int32 = 0, log: Bool = false) async throws {
        try await llama.loadModel(at: path, temperature: temperature, distribution: distribution, batchCapacity: batchCapacity, maxSequenceIdsPerToken: maxSequenceIdsPerToken, embeddingSize: embeddingSize)
    }
    
    /// Generates a complete response for the given prompt in a single operation.
    /// - Parameter prompt: The input text to send to the model.
    /// - Note: Calls `delegate.didGenerateResponse(_:)` or `delegate.generateResponseFailed(_:)`.
    public func promptGenerateResponse(prompt: String) async {
        await llama.promptGenerateResponse(prompt: prompt)
    }
    
    /// Runs the completion loop for the given prompt, streaming tokens one by one.
    /// - Parameter prompt: The input text to send to the model.
    /// - Note: Calls `delegate.getTokenFromCompletionLoop(_:)` as tokens are produced,
    ///   and `delegate.finishTokenFomCompletionLoop()` once generation ends.
    public func promptCompletionLoop(prompt: String) async {
        await llama.promptCompletionLoop(prompt: prompt)
    }
    
    /// Starts the completion loop with configurable generation length.
    /// - Parameter prompt: The input text to send to the model.
    /// - Note: Similar to `promptCompletionLoop`, but allows generation length to be adjusted.
    public func CompleteLoop(prompt: String, generationLength: Int32 = 128) async {
        await llama.CompleteLoop(prompt: prompt, generationLength: generationLength)
    }
    
    /// Generates a trimmed single response for the given prompt.
    /// - Parameter prompt: The input text to send to the model.
    /// - Note: The result is trimmed of whitespace and punctuation before being delivered
    ///   via `delegate.didGenerateResponse(_:)`.
    public func CompleteGenerateResponst(prompt: String, generationLength: Int32 = 128) async {
        await llama.CompleteGenerateResponst(prompt: prompt, generationLength: generationLength)
    }
    
    /// Retrieves the message log containing model load events and errors.
    /// - Returns: A string log of messages collected so far.
    public func getMessageLogs() -> String {
        return llama.messageLog
    }
    
#if DEBUG
    /// Runs a benchmark on the currently loaded model.
    /// - Note: Prints results to the console and passes them to `delegate.benchMarkMessage(_:)`.
    public func bench() async {
        await llama.bench()
        print(llama.messageLog)
    }
#endif

}
