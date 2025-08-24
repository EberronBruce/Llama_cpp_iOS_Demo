//
//  LlamaState.swift
//  llamaTest
//
//  Created by Bruce Burgess on 8/10/25.
//

import Foundation

struct Model: Identifiable {
    var id = UUID()
    var name: String
    var url: String
    var filename: String
    var status: String?
}

public protocol LlamaDelegate: AnyObject {
    func didGenerateResponse(_ response: String)
    func generateResponseFailed(_ error: Error)
    func getTokenFromCompletionLoop(_ token: String)
    func finishTokenFomCompletionLoop()
    func benchMarkMessage(_ message: String)
}


@MainActor
internal class LlamaState: NSObject {
    weak var delegate: LlamaDelegate?
    private(set) var isModelLoaded = false
    private(set) var isGeneratingResponse = false
    
    private var maxToken: Int = 128
    private var stopTokens: [String] = []
    private(set) var messageLog = ""
    private let NS_PER_S = 1_000_000_000.0
    
    @Published var isLoading: Bool = false

    private var llamaContext: LlamaContext?
    
    enum LoadError: Error, Equatable {
        case couldNotLocateModel
        case pathToModelEmpty
        case unableToLoadModel(String)
    }

    override init() {
        super.init()
    }
    
    
    func setStopTokens(tokens: [String]) {
        stopTokens = tokens
    }
    
    func setMaxToken(maxToken: Int) {
        self.maxToken = maxToken
    }
    
    
    func loadModel(at path: String, temperature: Float, distribution: UInt32, batchCapacity: Int32, maxSequenceIdsPerToken: Int32, embeddingSize: Int32, log: Bool = false,  completion: @escaping (Result<Void, Error>) -> Void) {
        if path.isEmpty {
            if log { self.messageLog += "No model path specified\n" }
            completion(.failure(LoadError.pathToModelEmpty))
            return
        }
        guard FileManager.default.fileExists(atPath: path) else {
            messageLog += "Model file not found at \(path)\n"
            completion(.failure(LoadError.couldNotLocateModel))
            return
        }
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let context = try LlamaContext.create_context(path: path, temperature: temperature, distribution: distribution, batchCapacity: batchCapacity, maxSquenceIdsPerToken: maxSequenceIdsPerToken, embeddingSize: embeddingSize)
                Task { @MainActor in
                    self.llamaContext = context
                    self.isModelLoaded = true
                    if log { self.messageLog += "Loaded model \(path)\n" }
                    completion(.success(()))
                }
            }catch {
                Task { @MainActor in
                    if log { self.messageLog += "\(error.localizedDescription)\n" }
                    completion(.failure(LoadError.unableToLoadModel(error.localizedDescription)))
                }
            }
        }
    }
    
    func loadModel(at path: String, temperature: Float, distribution: UInt32, batchCapacity: Int32, maxSequenceIdsPerToken: Int32, embeddingSize: Int32, log: Bool = false) async throws {
        try await withCheckedThrowingContinuation { continuation in
            loadModel(at: path, temperature: temperature, distribution: distribution, batchCapacity: batchCapacity, maxSequenceIdsPerToken: maxSequenceIdsPerToken, embeddingSize: embeddingSize, log: log) { result in
                switch result {
                case .success:
                    continuation.resume()
                case .failure(let error):
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    
    
    func promptGenerateResponse(prompt: String) async {
        guard let llamaContext else {
            return
        }
        do {
            isGeneratingResponse = true
            try await llamaContext.feedPrompt(prompt)
            let response = try await llamaContext.generateResponse(maxTokens: maxToken, stop: stopTokens)
            isGeneratingResponse = false
            delegate?.didGenerateResponse(response)
        } catch {
            delegate?.generateResponseFailed(error)
        }
    }
    
    func promptCompletionLoop(prompt: String) async {
        guard let llamaContext else {
            return
        }
        
        do {
            try await llamaContext.feedPrompt(prompt)

            while await !llamaContext.is_done {
                let result = try await llamaContext.completion_loop()
                if stopTokens.contains(where: { result.contains($0) }) {
                    await llamaContext.markDone()
                    break
                }
                delegate?.getTokenFromCompletionLoop(result)
            }
            delegate?.finishTokenFomCompletionLoop()
        } catch {
            delegate?.generateResponseFailed(error)
        }
    }
    
    func CompleteLoop(prompt: String) async {
        guard let llamaContext else {
            return
        }
        do {
            try await llamaContext.completion_init(text: prompt)
            while await !llamaContext.is_done {
                let result = try await llamaContext.completion_loop()
                if stopTokens.contains(where: { result.contains($0) }) {
                    await llamaContext.markDone()
                    break
                }
                delegate?.getTokenFromCompletionLoop(result)
            }
            delegate?.finishTokenFomCompletionLoop()
        } catch {
            delegate?.generateResponseFailed(error)
        }
    }
    
    func CompleteGenerateResponst(prompt: String) async {
        guard let llamaContext else {
            return
        }
        do {
            try await llamaContext.completion_init(text: prompt)
            let result = try await llamaContext.generateResponse(maxTokens: maxToken, stop: stopTokens)
            
            let trimmedResponse = result
                .trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
                .trimmingCharacters(in: CharacterSet.punctuationCharacters)
            delegate?.didGenerateResponse(trimmedResponse)
        } catch {
            delegate?.generateResponseFailed(error)
        }
    }


    func bench() async {
        guard let llamaContext else {
            return
        }

        print("\n")
        delegate?.benchMarkMessage("Running benchmark....\n")
        print("Running benchmark...\n")
        print("Model info: ")
        
        let modelInfo = await llamaContext.model_info()
        delegate?.benchMarkMessage("Model info: \(modelInfo)\n")
        print("\(modelInfo) \n")

        let t_start = DispatchTime.now().uptimeNanoseconds
        let _ = await llamaContext.bench(pp: 8, tg: 4, pl: 1) // heat up
        let t_end = DispatchTime.now().uptimeNanoseconds

        let t_heat = Double(t_end - t_start) / NS_PER_S
        delegate?.benchMarkMessage("Heat up time: \(t_heat) seconds, please wait...\n")
        print("Heat up time: \(t_heat) seconds, please wait...\n")

        // if more than 5 seconds, then we're probably running on a slow device
        if t_heat > 5.0 {
            delegate?.benchMarkMessage("Heat up time is too long, aborting benchmark\n")
            print("Heat up time is too long, aborting benchmark\n")
            return
        }

        let result = await llamaContext.bench(pp: 512, tg: 128, pl: 1, nr: 3)

        delegate?.benchMarkMessage(result)
        print(result)
        print("\n")
    }

    func clear() async {
        guard let llamaContext else {
            return
        }
        await llamaContext.clear()
    }
    
}
