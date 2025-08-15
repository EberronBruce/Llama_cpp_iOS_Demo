//
//  ContentView.swift
//  llamaTest
//
//  Created by Bruce Burgess on 8/14/25.
//

import SwiftUI

struct ContentView: View {
    @StateObject var llamaState = LlamaState()
    @State private var userMessage = ""

    var body: some View {
        NavigationView {
            VStack {
                messageList
                inputBar
            }
            .navigationTitle("Mimir")
            .toolbar {
                ToolbarItemGroup(placement: .navigationBarTrailing) {
                    NavigationLink(destination: ModelManagerView(llamaState: llamaState)) {
                                  Image(systemName: "gearshape")
                              }
                    Button("Bench") { bench() }
                    Button("Clear") { clear() }
                }
            }
        }
    }

    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    ForEach(llamaState.messages, id: \.id) { message in
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
            .onChange(of: llamaState.messages.count, initial: false) {
                guard let lastId = llamaState.messages.last?.id else { return }
                withAnimation { proxy.scrollTo(lastId, anchor: .bottom) }
            }
        }
    }

    private var inputBar: some View {
        HStack {
            TextField("Type your message...", text: $userMessage, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(5)

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
    }

    private func sendMessage() {
        let text = userMessage.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        
        hideKeyboard()
        let messageCopy = text
        userMessage = ""
        Task(priority: .userInitiated) {
            await llamaState.complete(text: messageCopy)
        }

    }

    private func bench() {
        Task { await llamaState.bench() }
    }

    private func clear() {
        Task { await llamaState.clear() }
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

#if canImport(UIKit)
extension View {
    func hideKeyboard() {
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    }
}
#endif

