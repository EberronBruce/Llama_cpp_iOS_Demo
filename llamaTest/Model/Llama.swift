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
    
    public init() {
        self.llama = LlamaState()
    }
    
    public weak var delegate: LlamaDelegate? {
        get { llama.delegate }
        set { llama.delegate = newValue }
    }
    
    public func isModelLoaded() -> Bool {
        llama.isModelLoaded
    }
    
    public func isGeneratingResponse() -> Bool {
        llama.isGeneratingResponse
    }
    
    public func setStopTokens(tokens: [String]) {
        llama.setStopTokens(tokens: tokens)
    }
    
    public func setMaxToken(maxToken: Int) {
        llama.setMaxToken(maxToken: maxToken)
    }
    
    public func clear() async {
        await llama.clear()
    }
    
    public func initializeModel(at path: String, temperature: Float = 0.5, distribution: UInt32 = 1234, batchCapacity: Int32 = 512, maxSequenceIdsPerToken: Int32 = 1, embeddingSize: Int32 = 0, log: Bool = false, completion: @escaping (Result<Void, Error>) -> Void) {
        llama.loadModel(at: path, temperature: temperature, distribution: distribution, batchCapacity: batchCapacity, maxSequenceIdsPerToken: maxSequenceIdsPerToken, embeddingSize: embeddingSize, log: log, completion: completion)
    }
    
    public func initializeModel(at path: String, temperature: Float = 0.5, distribution: UInt32 = 1234, batchCapacity: Int32 = 512, maxSequenceIdsPerToken: Int32 = 1, embeddingSize: Int32 = 0, log: Bool = false) async throws {
        try await llama.loadModel(at: path, temperature: temperature, distribution: distribution, batchCapacity: batchCapacity, maxSequenceIdsPerToken: maxSequenceIdsPerToken, embeddingSize: embeddingSize)
    }
    
    public func promptGenerateResponse(prompt: String) async {
        await llama.promptGenerateResponse(prompt: prompt)
    }
    
    public func promptCompletionLoop(prompt: String) async {
        await llama.promptCompletionLoop(prompt: prompt)
    }
    
    public func CompleteLoop(prompt: String) async {
        await llama.CompleteLoop(prompt: prompt)
    }
    
    public func CompleteGenerateResponst(prompt: String) async {
        await llama.CompleteGenerateResponst(prompt: prompt)
    }
    
    public func getMessageLogs() -> String {
        return llama.messageLog
    }
    
#if DEBUG
    public func bench() async {
        await llama.bench()
        print(llama.messageLog)
    }
#endif

}
