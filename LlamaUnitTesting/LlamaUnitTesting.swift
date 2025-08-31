//
//  LlamaUnitTesting.swift
//  LlamaUnitTesting
//
//  Created by Bruce Burgess on 8/23/25.
//

import Testing
import Foundation
@testable import LlamaCore
import UIKit

@MainActor
struct LlamaUnitTesting {
    @Test func testPromptGenerateResponseCallsDelegate() async throws {
        let mockContext = MockLlamaContext()
        mockContext.generatedResponse = "Hello World"

        let state = LlamaState(testContext: mockContext)
        let mockDelegate = MockDelegate()
        state.delegate = mockDelegate

        await state.promptGenerateResponse(prompt: "Test Prompt")

        #expect(mockDelegate.didGenerateResponseCalled == true)
        #expect(mockDelegate.lastResponse == "Hello World")
    }

    @Test func testCompletionLoopSendsTokens() async throws {
        let mockContext = MockLlamaContext()
        mockContext.completionLoopResults = ["Token1", "Token2"]

        let state = LlamaState(testContext: mockContext)
        let mockDelegate = MockDelegate()
        state.delegate = mockDelegate

        await state.promptCompletionLoop(prompt: "Test")

        #expect(mockDelegate.tokens.contains("Token1"))
        #expect(mockDelegate.tokens.contains("Token2"))
        #expect(mockDelegate.finishCalled == true)
    }
    
    @Test func testCompleteLoopSendsTokensAndFinishes() async throws {
        let mockContext = MockLlamaContext()
        mockContext.completionLoopResults = ["Loop1", "Loop2"]
        
        let state = LlamaState(testContext: mockContext)
        let mockDelegate = MockDelegate()
        state.delegate = mockDelegate
        
        await state.CompleteLoop(prompt: "Test Loop")
        
        #expect(mockDelegate.tokens.contains("Loop1"))
        #expect(mockDelegate.tokens.contains("Loop2"))
        #expect(mockDelegate.finishCalled == true)
    }
    
    @Test func testCompleteGenerateResponseCallsDelegate() async throws {
        let mockContext = MockLlamaContext()
        mockContext.generatedResponse = "Complete Response"
        
        let state = LlamaState(testContext: mockContext)
        let mockDelegate = MockDelegate()
        state.delegate = mockDelegate
        
        await state.CompleteGenerateResponst(prompt: "Test Prompt")
        
        #expect(mockDelegate.didGenerateResponseCalled == true)
        #expect(mockDelegate.lastResponse == "Complete Response")
    }

    @Test func testMemoryWarningTriggersClear() async throws {
        let mockContext = MockLlamaContext()
        let state = LlamaState(testContext: mockContext)
        let mockDelegate = MockDelegate()
        state.delegate = mockDelegate

        // Trigger memory warning via NotificationCenter
        await MainActor.run {
            NotificationCenter.default.post(name: UIApplication.didReceiveMemoryWarningNotification, object: nil)
        }

        // Check that the delegate got benchmark message
        #expect(mockDelegate.benchMessages.contains(where: { $0.contains("Memory warning") }))
    }

    @Test func testBenchSendsMessages() async throws {
        let mockContext = MockLlamaContext()
        let state = LlamaState(testContext: mockContext)
        let mockDelegate = MockDelegate()
        state.delegate = mockDelegate
        
        await state.bench()
        
        #expect(mockDelegate.benchMessages.contains(where: { $0.contains("Running benchmark") }))
        #expect(mockDelegate.benchMessages.contains(where: { $0.contains("mock benchmark") }))
    }

    @Test func testClearResetsContext() async throws {
        let mockContext = MockLlamaContext()
        mockContext.fedPrompt = "Some prompt"
        
        let state = LlamaState(testContext: mockContext)
        await state.clear()
        
        #expect(mockContext.fedPrompt == nil)
    }


}


class MockDelegate: LlamaDelegate {
    func didRecieveMemoryWarning() {
        benchMessages.append("Memory warning!")
    }
    
    var didGenerateResponseCalled = false
    var lastResponse: String?
    var generateResponseFailedCalled = false
    var lastError: Error?
    var tokens: [String] = []
    var finishCalled = false
    var benchMessages: [String] = []
    
    func didGenerateResponse(_ response: String) {
        didGenerateResponseCalled = true
        lastResponse = response
    }
    
    func generateResponseFailed(_ error: Error) {
        generateResponseFailedCalled = true
        lastError = error
    }
    
    func getTokenFromCompletionLoop(_ token: String) {
        tokens.append(token)
    }
    
    func finishTokenFomCompletionLoop() {
        finishCalled = true
    }
    
    func benchMarkMessage(_ message: String) {
        benchMessages.append(message)
    }
}


class MockLlamaContext: LlamaContextProtocol {
    var is_done: Bool = false
    var n_cur = 0
    
    var fedPrompt: String?
    var generatedResponse: String = "mocked response"
    var completionLoopResults: [String] = []
    private var index = 0

    func nextToken() -> String? {
        guard index < completionLoopResults.count else { return nil }
        defer { index += 1 }
        return completionLoopResults[index]
    }
    
    func model_info() async -> String {
        return "Mock Model v1.0"
    }
    
    func markDone() async {
        is_done = true
    }
    
    func resetDone() async {
        is_done = false
    }
    
    func clear() async {
        fedPrompt = nil
    }
    
    func get_n_tokens() async -> Int32 {
        return 42
    }
    
    func feedPrompt(_ text: String) async throws {
        fedPrompt = text
    }
    
    func generateResponse(maxTokens: Int, stop: [String]) async throws -> String {
        return generatedResponse
    }
    
    func completion_init(text: String, generationLength: Int32) async throws {
        fedPrompt = text
    }
    
    func completion_loop() async throws -> String {
        guard !is_done else {
            return ""
        }

        if n_cur < completionLoopResults.count {
            let token = completionLoopResults[n_cur]
            n_cur += 1
            return token
        } else {
            is_done = true
            return "" // mimic the behavior when LLM is finished
        }
    }
    
    func bench(pp: Int, tg: Int, pl: Int, nr: Int) async -> String {
        return "mock benchmark"
    }
}


