//
//  ModelManagerView.swift
//  llamaTest
//
//  Created by Bruce Burgess on 8/14/25.
//
import SwiftUI

struct ModelManagerView: View {
    @ObservedObject var llamaState: LlamaState
    @State private var showingHelp = false

    func delete(at offsets: IndexSet) {
        offsets.forEach { offset in
            let model = llamaState.downloadedModels[offset]
            let fileURL = getDocumentsDirectory().appendingPathComponent(model.filename)
            do {
                try FileManager.default.removeItem(at: fileURL)
            } catch {
                print("Error deleting file: \(error)")
            }
        }
        llamaState.downloadedModels.remove(atOffsets: offsets)
    }

    func getDocumentsDirectory() -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    var body: some View {
        List {
            Section(header: Text("Download Models From Hugging Face")) {
                HStack {
                    InputButton(llamaState: llamaState)
                }
            }
            Section(header: Text("Downloaded Models")) {
                ForEach(llamaState.downloadedModels) { model in
                    DownloadButton(
                        llamaState: llamaState,
                        modelName: model.name,
                        modelUrl: model.url,
                        filename: model.filename
                    )
                }
                .onDelete(perform: delete)
            }
            Section(header: Text("Default Models")) {
                ForEach(llamaState.undownloadedModels) { model in
                    DownloadButton(
                        llamaState: llamaState,
                        modelName: model.name,
                        modelUrl: model.url,
                        filename: model.filename
                    )
                }
            }
        }
        .listStyle(GroupedListStyle())
        .navigationTitle("Model Settings")
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                Button("Help") { showingHelp = true }
            }
        }
        .sheet(isPresented: $showingHelp) {
            NavigationView {
                VStack(alignment: .leading) {
                    Text("1. Make sure the model is in GGUF Format")
                        .padding()
                    Text("2. Copy the download link of the quantized model")
                        .padding()
                    Spacer()
                }
                .navigationTitle("Help")
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .navigationBarTrailing) {
                        Button("Done") { showingHelp = false }
                    }
                }
            }
        }
    }
}
