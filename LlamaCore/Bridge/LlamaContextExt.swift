//
//  LlamaContextExt.swift
//  llamaTest
//
//  Created by Bruce Burgess on 8/23/25.
//
import Foundation


internal protocol LlamaContextProtocol {
    func model_info() async -> String
    
    // ---- Prompt / Response ----
    func feedPrompt(_ text: String) async throws
    func generateResponse(maxTokens: Int, stop: [String]) async throws -> String
    
    // ---- Completion APIs ----
    func completion_init(text: String, generationLength: Int32) async throws
    func completion_loop() async throws -> String
    
    // ---- Bench ----
    func bench(pp: Int, tg: Int, pl: Int, nr: Int) async -> String
    func markDone() async
    func resetDone() async
    func clear() async
    func get_n_tokens() async -> Int32
    var is_done: Bool { get async}
}

extension LlamaContext: LlamaContextProtocol {}
