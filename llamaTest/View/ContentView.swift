//
//  ContentView.swift
//  llamaTest
//
//  Created by Bruce Burgess on 8/14/25.
//

import SwiftUI

struct ContentView: View {
    @StateObject var bridge = LlamaBridge()
    
    private var llama = Llama()
    @State private var userMessage = ""
    @State private var llamaResponse = ""
    
    let systemPrompt:String = "You are a polite assistant. Only reply with a short, standalone response to the user’s input. Do not give examples, stories, exercises, invented content, or continue incomplete sentences. For greetings like 'Hello', respond with a greeting and optional follow-up question like 'How can I help you?'."
    
    init() {        
        self.userMessage = userMessage
    }

    var body: some View {
        NavigationView {
            ZStack{
                VStack {
                    messageList
                    inputBar
                }
                .navigationTitle("Mimir")
                .toolbar {
                    ToolbarItemGroup(placement: .navigationBarTrailing) {
                        //                    NavigationLink(destination: ModelManagerView(llamaState: llamaState)) {
                        //                                  Image(systemName: "gearshape")
                        //                              }
                        Button("Bench") { bench() }
                        Button("Clear") { clear() }
                    }
                }
                
                if bridge.isLoading {
                       Color.black.opacity(0.4) // dark transparent background
                           .ignoresSafeArea()

                       ProgressView()
                           .progressViewStyle(CircularProgressViewStyle(tint: .white))
                           .scaleEffect(2) // bigger spinner
                   }
            }
            
        }
    }


    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    ForEach(bridge.messages, id: \.id) { message in
                        ChatBubble(message: message)
                            .id(message.id)
                    }
                }
                .padding()
            }
            .contentShape(Rectangle()) // makes the whole area tappable
            .onTapGesture {
                hideKeyboard()
            }
            .onChange(of: bridge.messages.count, initial: false) {
                guard let lastId = bridge.messages.last?.id else { return }
                withAnimation { proxy.scrollTo(lastId, anchor: .bottom) }
            }
        }
    }

    private var inputBar: some View {
        HStack {
            TextField("Type your message...", text: $userMessage, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(5)
                .disabled(bridge.isLoading)

            Button(action: sendMessage) {
                Image(systemName: "paperplane.fill")
                    .font(.system(size: 18))
                    .padding(8)
            }
            .buttonStyle(.borderedProminent)
            .disabled(userMessage.isEmpty)
        }
        .padding()
        .background(.ultraThinMaterial)
        .onAppear() {
            Task {
                do{
                    bridge.isLoading = true
                    bridge.messages.append(ChatMessage(text: "Loading model...", isUser: false))
                    let modelPath = Bundle.main.path(forResource: "phi-2-q4_0", ofType: "gguf")!
                    try await self.llama.initializeModel(at: modelPath, temperature: 0.5, distribution: 1234, batchCapacity: 512, maxSequenceIdsPerToken: 1, embeddingSize: 0)
//                    try await self.llamaState.loadModel(at: modelPath, temperature: 0.5, distribution: 1234, batchCapacity: 512, maxSequenceIdsPerToken: 1, embeddingSize: 0)
                    bridge.isLoading = false
                    let model = extractModelName(from: modelPath)
                    bridge.messages.append(ChatMessage(text: "Model: \(model) Loaded", isUser: false))
                    self.llama.setStopTokens(tokens: ["<END>", "User:"])
                    self.llama.setMaxToken(maxToken: 512)
                } catch {
                    bridge.messages.append(ChatMessage(text: error.localizedDescription, isUser: false))
                }
            }
            llama.delegate = bridge
//            llamaState.setSystemPrompt(text: systemPrompt)
        }
    }

    private func sendMessage() {
        let text = userMessage.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        
        hideKeyboard()
        let messageCopy = text
        bridge.messages.append(ChatMessage(text: messageCopy, isUser: true))
        userMessage = ""
        Task(priority: .userInitiated) {
            bridge.isLoading = true
            let fullPrompt = systemPrompt + "User:" + messageCopy + ".<END>"
            await llama.promptGenerateResponse(prompt: fullPrompt)
        }

    }

    private func bench() {
        Task {
            bridge.isLoading = true
            await llama.bench()
            bridge.isLoading = false
        }
    }

    private func clear() {
        Task {
            bridge.isLoading = true
            await llama.clear()
            bridge.isLoading = false
        }
    }
}


struct ChatBubble: View {
    var message: ChatMessage
    
    var body: some View {
        HStack {
            if message.isUser {
                Spacer()
                Text(message.text)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding()
                    .background(Color.blue.opacity(0.8))
                    .foregroundColor(.white)
                    .cornerRadius(12)
            } else {
                Text(message.text)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding()
                    .background(Color.gray.opacity(0.2))
                    .foregroundColor(.primary)
                    .cornerRadius(12)
                Spacer()
            }
        }
    }
}

class LlamaBridge: ObservableObject, LlamaDelegate {
    @Published var messages : [ChatMessage] = []
    @Published var isLoading: Bool = false
    func didGenerateResponse(_ response: String) {
        print("LLM Response: \(response)")
        let trimmedResponse = response
            .trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
//            .trimmingCharacters(in: CharacterSet.punctuationCharacters)
            .replacingOccurrences(of: "AI: ", with: "")
            .replacingOccurrences(of: "Assistant: ", with: "")
            .replacingOccurrences(of: "Assistant:", with: "")
            .replacingOccurrences(of: "AI:", with: "")
        messages.append(ChatMessage(text: trimmedResponse, isUser: false))
        isLoading = false
    }
    
    func generateResponseFailed(_ error: any Error) {
        messages.append(ChatMessage(text: error.localizedDescription, isUser: false))
        isLoading = false
    }
    
    func getTokenFromCompletionLoop(_ token: String) {
        messages.append(ChatMessage(text: token, isUser: false))
    }
    
    func finishTokenFomCompletionLoop() {
        isLoading = false
    }
    
    func benchMarkMessage(_ message: String) {
        messages.append(ChatMessage(text: message, isUser: false))
    }
    

}

#if canImport(UIKit)
extension View {
    func hideKeyboard() {
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    }
}
#endif

func extractModelName(from path: String) -> String {
    let url = URL(fileURLWithPath: path)
    let fileName = url.deletingPathExtension().lastPathComponent
    return fileName
}
