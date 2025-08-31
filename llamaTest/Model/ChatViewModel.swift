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



