//
//  ChatViewModel.swift
//  llamaTest
//
//  Created by Bruce Burgess on 8/14/25.
//
import Foundation
import SwiftUI

struct ChatMessage: Identifiable {
    let id = UUID()
    let text: String
    let isUser: Bool
}


//class ChatViewModel: ObservableObject {
//    @Published var messages: [Message] = []
//    
//    func addUserMessage(_ text: String) {
//        messages.append(Message(role: "user", text: text))
//    }
//    
//    func addAssistantMessage(_ text: String) {
//        messages.append(Message(role: "assistant", text: text))
//    }
//}
