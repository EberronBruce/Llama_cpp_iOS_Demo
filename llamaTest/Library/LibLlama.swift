import Foundation
import llama
import MachO
import Darwin

let isDebugMode = true

enum LlamaError: Error {
    case couldNotInitializeContext
    case batchCapacityExceeded
}

actor LlamaContext {
    let LLAMA_INVALID_TOKEN: llama_token = -1
    // You may need this constant, as Swift doesn’t always pull it from C headers
    private static let HOST_VM_INFO64_COUNT = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)

    let batchCapacity: Int32
    let embeddingSize: Int32
    let maxSeqIdsPerToken: Int32
    private var model: OpaquePointer
    private var context: OpaquePointer
    private var vocab: OpaquePointer
    private var sampling: UnsafeMutablePointer<llama_sampler>
    // Owns allocated batch buffers; must be freed exactly once in deinit.
    // Never reassign batch without freeing first.
    private var batch: llama_batch
    private var tokens_list: [llama_token]
    var is_done: Bool = false

    // Holds partial UTF-8 bytes until we have enough to decode a full character
    private var _tempData: Data? = nil
    
    private static var backendInitialized = false

    var n_len: Int32 = 1024
    var n_cur: Int32 = 0

    var n_decode: Int32 = 0

    init(model: OpaquePointer, context: OpaquePointer) {
        self.batchCapacity = 512
        self.embeddingSize = 0
        self.maxSeqIdsPerToken = 1
        self.model = model
        self.context = context
        self.tokens_list = []
        
        let rawBatch = llama_batch_init(batchCapacity, embeddingSize, maxSeqIdsPerToken)
        guard rawBatch.token != nil,
              rawBatch.pos != nil,
              rawBatch.n_seq_id != nil,
              rawBatch.seq_id != nil,
              rawBatch.logits != nil else {
            fatalError("llama_batch_init returned invalid pointers")
        }
        self.batch = rawBatch
        
        self._tempData = nil
        let sparams = llama_sampler_chain_default_params()
        self.sampling = llama_sampler_chain_init(sparams)
        
        if isDebugMode {
            llama_sampler_chain_add(self.sampling, llama_sampler_init_temp(0.0))
            llama_sampler_chain_add(self.sampling, llama_sampler_init_dist(0))
        } else {
            llama_sampler_chain_add(self.sampling, llama_sampler_init_temp(0.4))
            llama_sampler_chain_add(self.sampling, llama_sampler_init_dist(1234))
        }
        
        vocab = llama_model_get_vocab(model)
        
        print("Model, context, and vocab initialized successfully")
    }

    deinit {
        llama_sampler_free(sampling)
        llama_batch_free(batch)
        llama_free(context)
        llama_model_free(model)
        llama_backend_free()
    }
    
    // Returns available memory in bytes
    static func getAvailableMemory() -> UInt64 {
        if #available(iOS 18, *) {
            return UInt64(os_proc_available_memory())

        } else {
            var stats = vm_statistics64()
            var count = HOST_VM_INFO64_COUNT

            let result = withUnsafeMutablePointer(to: &stats) {
                $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                    host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
                }
            }

            if result == KERN_SUCCESS {
                let freeBytes = UInt64(stats.free_count) * UInt64(vm_kernel_page_size)
                let inactiveBytes = UInt64(stats.inactive_count) * UInt64(vm_kernel_page_size)
                // Available = free + inactive memory
                return freeBytes &+ inactiveBytes
            } else {
                print("[WARN] Could not get available memory")
                return 0
            }
        }
    }
    
    
    static func create_context(path: String) async throws -> LlamaContext {
        if !backendInitialized {
            llama_backend_init()
            backendInitialized = true
        }

        // Detect RAM budget before trying to load
        let availableMem = getAvailableMemory()
        print("[INFO] Available memory: \(availableMem / 1024 / 1024) MB")

        // Make a starting guess based on memory
        var start_n_ctx: Int32
        var start_n_gpu_layers: Int32
        let hadSavedConfig: Bool

        if let savedLayers = UserDefaults.standard.value(forKey: "lastWorkingGpuLayers") as? Int32,
           let savedCtx = UserDefaults.standard.value(forKey: "lastWorkingCtx") as? Int32 {
            print("[INFO] Using saved config: n_ctx=\(savedCtx), n_gpu_layers=\(savedLayers)")
            start_n_ctx = savedCtx
            start_n_gpu_layers = savedLayers
            hadSavedConfig = true
        } else {
            if availableMem < 2_000_000_000 {
                start_n_ctx = 512
                start_n_gpu_layers = 16
            } else if availableMem < 4_000_000_000 {
                start_n_ctx = 1024
                start_n_gpu_layers = 64
            } else {
                start_n_ctx = 2048
                start_n_gpu_layers = 999
            }
            hadSavedConfig = false
        }

        print("[INFO] Starting guess: n_ctx=\(start_n_ctx), n_gpu_layers=\(start_n_gpu_layers)")

        var model_params = llama_model_default_params()
        var model: OpaquePointer? = nil

    #if targetEnvironment(simulator)
        // Simulator never has GPU acceleration
        model_params.n_gpu_layers = 0
        print("[INFO] Running in simulator. Forcing n_gpu_layers=0.")
        guard let m = llama_model_load_from_file(path, model_params) else {
            throw LlamaError.couldNotInitializeContext
        }
        model = m
    #else
        // On device: optionally try FULL Metal push first (only if no saved config).
        if !hadSavedConfig {
            print("[INFO] Attempting full Metal load (all layers on GPU first try)…")
            
            // Try full offload
            model_params.n_gpu_layers = Int32.max
            if let m = llama_model_load_from_file(path, model_params) {
                print("[SUCCESS] Full Metal load succeeded with n_gpu_layers=\(model_params.n_gpu_layers).")
                model = m
                UserDefaults.standard.set(model_params.n_gpu_layers, forKey: "lastWorkingGpuLayers")
                UserDefaults.standard.set(start_n_ctx, forKey: "lastWorkingCtx")
            } else {
                print("[FAIL] Full Metal load failed. Falling back to CPU-only…")
                model_params.n_gpu_layers = 0 // CPU fallback
                guard let cpuModel = llama_model_load_from_file(path, model_params) else {
                    throw LlamaError.couldNotInitializeContext
                }
                model = cpuModel
            }
        }

        // If full push didn’t work (or we had a saved config), try descending lists.
        if model == nil {
            var tryGpuLayersList: [Int32] = []

            // When no saved config, prepend a big first try (if we didn’t already)
            if !hadSavedConfig {
                tryGpuLayersList.append(1024)
            }

            if start_n_gpu_layers >= 999 {
                tryGpuLayersList += [512, 256, 128, 64, 32, 16, 8, 0]
            } else if start_n_gpu_layers >= 64 {
                tryGpuLayersList += [start_n_gpu_layers, 32, 16, 8, 0]
            } else {
                tryGpuLayersList += [start_n_gpu_layers, 8, 0]
            }

            // Deduplicate while preserving order
            var seen = Set<Int32>()
            tryGpuLayersList = tryGpuLayersList.filter { seen.insert($0).inserted }

            for candidate in tryGpuLayersList {
                model_params.n_gpu_layers = candidate
                print("[INFO] Attempting model load with n_gpu_layers=\(candidate)…")
                if let m = llama_model_load_from_file(path, model_params) {
                    print("[SUCCESS] Model loaded with n_gpu_layers=\(candidate).")
                    model = m
                    UserDefaults.standard.set(candidate, forKey: "lastWorkingGpuLayers")
                    UserDefaults.standard.set(start_n_ctx, forKey: "lastWorkingCtx")
                    break
                } else {
                    print("[FAIL] Could not load with n_gpu_layers=\(candidate). Trying next…")
                }
            }
        }

        // If no model loaded at all, explicit CPU-only attempt (safety net)
        if model == nil {
            print("[WARN] GPU attempts failed. Trying CPU-only fallback.")
            model_params.n_gpu_layers = 0
            if let m = llama_model_load_from_file(path, model_params) {
                print("[SUCCESS] CPU-only model load succeeded.")
                model = m
                UserDefaults.standard.set(0, forKey: "lastWorkingGpuLayers")
                UserDefaults.standard.set(start_n_ctx, forKey: "lastWorkingCtx")
            } else {
                print("[ERROR] CPU-only model load also failed.")
                throw LlamaError.couldNotInitializeContext
            }
        }
    #endif

        guard let modelFinal = model else {
            throw LlamaError.couldNotInitializeContext
        }

        // Proceed to init context
        let n_threads = max(1, min(8, ProcessInfo.processInfo.processorCount - 2))
        var ctx_params = llama_context_default_params()
        ctx_params.n_ctx = UInt32(start_n_ctx)
        ctx_params.n_threads = Int32(n_threads)
        ctx_params.n_threads_batch = Int32(n_threads)

        // Context init + CPU-only retry if Metal backend fails at this stage
        guard let context = llama_init_from_model(modelFinal, ctx_params) else {
            llama_model_free(modelFinal)
            print("[WARN] Context init failed with GPU layers \(model_params.n_gpu_layers), retrying CPU-only…")

            model_params.n_gpu_layers = 0
            guard let cpuModel = llama_model_load_from_file(path, model_params) else {
                print("[ERROR] CPU-only model load failed.")
                throw LlamaError.couldNotInitializeContext
            }
            guard let cpuContext = llama_init_from_model(cpuModel, ctx_params) else {
                llama_model_free(cpuModel)
                print("[ERROR] CPU-only context init failed.")
                throw LlamaError.couldNotInitializeContext
            }

            let cpuLlamaContext = LlamaContext(model: cpuModel, context: cpuContext)
            let info = await cpuLlamaContext.model_info()
            print("[INFO] Loaded model description (CPU-only fallback): \(info)")
            return cpuLlamaContext
        }

        let llamaContext = LlamaContext(model: modelFinal, context: context)
        let info = await llamaContext.model_info()
        print("[INFO] Loaded model description: \(info)")
        return llamaContext
    }

    func model_info() async -> String {
        let capacity = 256
        let result = UnsafeMutablePointer<Int8>.allocate(capacity: capacity)
        result.initialize(repeating: Int8(0), count: capacity)
        defer {
            result.deinitialize(count: capacity)
            result.deallocate()
        }

        let nChars = llama_model_desc(model, result, capacity)
        guard nChars > 0 else {
            return "[Failed to get model description]"
        }

        return String(cString: result)
    }

    func get_n_tokens() -> Int32 {
        return batch.n_tokens;
    }

    func completion_init(text: String) {
        print("attempting to complete \"\(text)\"")

        tokens_list = tokenize(text: text, add_bos: true)
        defer { _tempData = nil }

        let n_ctx = llama_n_ctx(context)
        let n_kv_req = tokens_list.count + (Int(n_len) - tokens_list.count)

        print("\n n_len = \(n_len), n_ctx = \(n_ctx), n_kv_req = \(n_kv_req)")

        if n_kv_req > n_ctx {
            print("error: n_kv_req > n_ctx, the required KV cache size is not big enough")
        }

        for id in tokens_list {
            let cchars = token_to_piece(token: id)
            if let tokenString = String(bytes: cchars.map { UInt8(bitPattern: $0) }, encoding: .utf8) {
                print(tokenString)
            } else {
                print("[Invalid UTF8 token]")
            }
        }

        // Clear batch tokens count without freeing batch buffers to reuse allocated memory
        llama_batch_clear(&batch)

        do {
            for i1 in 0..<tokens_list.count {
                let i = Int(i1)
                try llama_batch_add(&batch, tokens_list[i], Int32(i), [0], false, maxCapacity: batchCapacity)
            }
        } catch {
            print("Error adding tokens to batch: \(error)")
            defer { _tempData = nil }
            return
        }
        batch.logits[Int(batch.n_tokens) - 1] = 1 // true

        let decodeResult = llama_decode(context, batch)
        if decodeResult != 0 {
            if batch.n_tokens > 0 && !is_done {
                print("Error: llama_decode failed with code \(decodeResult)")
                defer { _tempData = nil }
                is_done = true
                return
            } else {
                print("Warning: llama_decode returned \(decodeResult), no tokens or already done, continuing")
            }
        }

        n_cur = batch.n_tokens
    }

    func completion_loop() -> String {
        var new_token_id: llama_token = 0
        
        guard batch.n_tokens > 0 else {
            print("Error: batch has no tokens for sampling")
            is_done = true
            return ""
        }

        new_token_id = safeSamplerSample(tokenIndex: batch.n_tokens - 1)
        guard new_token_id != LLAMA_INVALID_TOKEN else {
            print("Error: llama_sampler_sample failed or returned invalid token")
            is_done = true
            return ""
        }

        if llama_vocab_is_eog(vocab, new_token_id) || n_cur == n_len {
            print("\n")
            is_done = true

            if let tempData = _tempData, let decoded = String(data: tempData, encoding: .utf8) {
                defer { _tempData = nil }
                return decoded
            } else {
                print("Warning: _tempData contained invalid UTF-8")
                defer { _tempData = nil }
                return ""
            }
        }

        let new_token_cchars = token_to_piece(token: new_token_id)
        let new_token_str = appendBytesAndExtractString(new_token_cchars)
        
        print(new_token_str)
        
        // Clear batch tokens count without freeing batch buffers to reuse allocated memory
        llama_batch_clear(&batch)
        
        do{
            try llama_batch_add(&batch, new_token_id, n_cur, [0], true, maxCapacity: batchCapacity)
        } catch {
            print("Error adding tokens to batch: \(error)")
            return "Error adding tokens to batch: \(error)"
        }
        

        n_decode += 1
        n_cur    += 1
        
        let decodeResult = llama_decode(context, batch)
        if decodeResult != 0 {
            if batch.n_tokens > 0 && !is_done {
                print("Error: llama_decode failed with code \(decodeResult)")
                defer { _tempData = nil }
                is_done = true
                return "Error: llama_decode failed with code \(decodeResult)"
            } else {
                print("Warning: llama_decode returned \(decodeResult), no tokens or already done, continuing")
            }
        }

        return new_token_str
    }

    func bench(pp: Int, tg: Int, pl: Int, nr: Int = 1) async -> String {
        var pp_avg: Double = 0
        var tg_avg: Double = 0

        var pp_std: Double = 0
        var tg_std: Double = 0

        for _ in 0..<nr {
            // bench prompt processing
            
            // Clear batch tokens count without freeing batch buffers to reuse allocated memory
            llama_batch_clear(&batch)
            
            do {
                let n_tokens = pp
                
                for i in 0..<n_tokens {
                    try llama_batch_add(&batch, 0, Int32(i), [0], false, maxCapacity: batchCapacity)
                }
                batch.logits[Int(batch.n_tokens) - 1] = 1 // true
            } catch {
                print("Error adding tokens to batch: \(error)")
                return "Error adding tokens to batch: \(error)"
            }
            
            
            llama_memory_clear(llama_get_memory(context), false)
            
            let t_pp_start = DispatchTime.now().uptimeNanoseconds / 1000;
            
            guard llama_decode(context, batch) == 0 else {
                print("Error: llama_decode failed during prompt")
                is_done = true
                return "Error: llama_decode failed during prompt"
            }
        
            
            llama_synchronize(context)

            let t_pp_end = DispatchTime.now().uptimeNanoseconds / 1000;

            // bench text generation

            llama_memory_clear(llama_get_memory(context), false)

            let t_tg_start = DispatchTime.now().uptimeNanoseconds / 1000;

            do {
                for i in 0..<tg {
                    
                    // Clear batch tokens count without freeing batch buffers to reuse allocated memory
                    llama_batch_clear(&batch)

                    for j in 0..<pl {
                        try llama_batch_add(&batch, 0, Int32(i), [Int32(j)], true, maxCapacity: batchCapacity)
                    }

                    guard llama_decode(context, batch) == 0 else {
                        print("Error: llama_decode failed during text generation")
                        is_done = true
                        return "Error: llama_decode failed during text generation"
                    }
                    llama_synchronize(context)
                }
            } catch {
                print("Error adding tokens to batch: \(error)")
                return "Error adding tokens to batch: \(error)"
            }


            let t_tg_end = DispatchTime.now().uptimeNanoseconds / 1000;

            llama_memory_clear(llama_get_memory(context), false)

            let t_pp = Double(t_pp_end - t_pp_start) / 1000000.0
            let t_tg = Double(t_tg_end - t_tg_start) / 1000000.0

            let speed_pp = Double(pp)    / t_pp
            let speed_tg = Double(pl*tg) / t_tg

            pp_avg += speed_pp
            tg_avg += speed_tg

            pp_std += speed_pp * speed_pp
            tg_std += speed_tg * speed_tg

            print("pp \(speed_pp) t/s, tg \(speed_tg) t/s")
        }

        pp_avg /= Double(nr)
        tg_avg /= Double(nr)

        if nr > 1 {
            pp_std = sqrt(pp_std / Double(nr - 1) - pp_avg * pp_avg * Double(nr) / Double(nr - 1))
            tg_std = sqrt(tg_std / Double(nr - 1) - tg_avg * tg_avg * Double(nr) / Double(nr - 1))
        } else {
            pp_std = 0
            tg_std = 0
        }

        let model_desc     = await model_info();
        let model_size     = String(format: "%.2f GiB", Double(llama_model_size(model)) / 1024.0 / 1024.0 / 1024.0);
        let model_n_params = String(format: "%.2f B", Double(llama_model_n_params(model)) / 1e9);
        let backend        = "Metal";
        let pp_avg_str     = String(format: "%.2f", pp_avg);
        let tg_avg_str     = String(format: "%.2f", tg_avg);
        let pp_std_str     = String(format: "%.2f", pp_std);
        let tg_std_str     = String(format: "%.2f", tg_std);

        var result = ""

        result += String("| model | size | params | backend | test | t/s |\n")
        result += String("| --- | --- | --- | --- | --- | --- |\n")
        result += String("| \(model_desc) | \(model_size) | \(model_n_params) | \(backend) | pp \(pp) | \(pp_avg_str) ± \(pp_std_str) |\n")
        result += String("| \(model_desc) | \(model_size) | \(model_n_params) | \(backend) | tg \(tg) | \(tg_avg_str) ± \(tg_std_str) |\n")

        return result;
    }

    func clear() {
        tokens_list.removeAll()
        _tempData = nil
        llama_memory_clear(llama_get_memory(context), true)
    }
    
    private func tokenize(text: String, add_bos: Bool) -> [llama_token] {
        let utf8Count = text.utf8.count
//        let n_tokens = utf8Count + (add_bos ? 1 : 0) + 1
        let n_tokens = max(utf8Count * 2, utf8Count + (add_bos ? 1 : 0) + 1)
        let tokens = UnsafeMutablePointer<llama_token>.allocate(capacity: n_tokens)
        defer {
            tokens.deallocate()
        }
        let tokenCount = llama_tokenize(vocab, text, Int32(utf8Count), tokens, Int32(n_tokens), add_bos, false)
        print("tokenizeSimple tokenCount: \(tokenCount)")
        guard tokenCount >= 0 else {
            print("Error: tokenizeSimple llama_tokenize failed with tokenCount \(tokenCount)")
            return []
        }
        return Array(UnsafeBufferPointer(start: tokens, count: Int(tokenCount)))
    }

    
    private func token_to_piece(token: llama_token) -> [CChar] {
        let initialCapacity = 8
        let result = UnsafeMutablePointer<Int8>.allocate(capacity: initialCapacity)
        result.initialize(repeating: 0, count: initialCapacity)
        defer {
            result.deinitialize(count: initialCapacity)
            result.deallocate()
        }
        let nTokens = llama_token_to_piece(vocab, token, result, Int32(initialCapacity), 0, false)

        if nTokens < 0 {
            let newCapacity = Int(-nTokens) + 1
            guard newCapacity > 0 else {
                print("Error: llama_token_to_piece returned invalid negative token count \(nTokens)")
                return []
            }
            let newResult = UnsafeMutablePointer<Int8>.allocate(capacity: newCapacity)
            newResult.initialize(repeating: 0, count: newCapacity)
            defer {
                newResult.deinitialize(count: newCapacity)
                newResult.deallocate()
            }
            let nNewTokens = llama_token_to_piece(vocab, token, newResult, Int32(newCapacity), 0, false)
            guard nNewTokens > 0 else {
                print("Error: llama_token_to_piece failed on second attempt with \(nNewTokens)")
                return []
            }
            newResult[Int(nNewTokens)] = 0
            return Array(UnsafeBufferPointer(start: newResult, count: Int(nNewTokens)))
        } else if nTokens > 0 {
            return Array(UnsafeBufferPointer(start: result, count: Int(nTokens)))
        } else {
            print("Error: llama_token_to_piece returned zero or invalid token length \(nTokens)")
            return []
        }
    }
    
    
    private func llama_batch_add(_ batch: inout llama_batch,
                                 _ id: llama_token,
                                 _ pos: llama_pos,
                                 _ seq_ids: [llama_seq_id],
                                 _ logits: Bool,
                                 maxCapacity: Int32) throws {

        let writeIndex = Int(batch.n_tokens)
        guard writeIndex < Int(maxCapacity) else {
            print("Batch capacity exceeded! \(writeIndex) >= \(maxCapacity)")
            throw LlamaError.batchCapacityExceeded
        }

        // token pointer -> buffer
        if let tokenPtr = batch.token {
            let tokenBuf = UnsafeMutableBufferPointer(start: tokenPtr, count: Int(maxCapacity))
            tokenBuf[writeIndex] = id
        } else {
            fatalError("batch.token is nil")
        }

        // pos
        if let posPtr = batch.pos {
            let posBuf = UnsafeMutableBufferPointer(start: posPtr, count: Int(maxCapacity))
            posBuf[writeIndex] = pos
        } else {
            fatalError("batch.pos is nil")
        }

        // n_seq_id
        if let nSeqPtr = batch.n_seq_id {
            let nSeqBuf = UnsafeMutableBufferPointer(start: nSeqPtr, count: Int(maxCapacity))
            nSeqBuf[writeIndex] = Int32(min(seq_ids.count, Int(maxSeqIdsPerToken)))
        } else {
            fatalError("batch.n_seq_id is nil")
        }

        // seq_id is an array of pointers; each pointer points to an array of length maxSeqIdsPerToken
        if let seqIdArrayPtr = batch.seq_id {
            // seqIdArrayPtr is UnsafeMutablePointer<UnsafeMutablePointer<llama_seq_id>?>?
            let seqIdPtrBuf = UnsafeMutableBufferPointer(start: seqIdArrayPtr, count: Int(maxCapacity))
            if let perTokenPtr = seqIdPtrBuf[writeIndex] {
                let perTokenBuf = UnsafeMutableBufferPointer(start: perTokenPtr, count: Int(maxSeqIdsPerToken))
                for i in 0..<min(seq_ids.count, Int(maxSeqIdsPerToken)) {
                    perTokenBuf[i] = seq_ids[i]
                }
            } else {
                // seq_id pointer for this token is nil — warn but continue
                print("Warning: seq_id[\(writeIndex)] is nil")
            }
        } else {
            fatalError("batch.seq_id is nil")
        }

        // logits
        if let logitsPtr = batch.logits {
            let logitsBuf = UnsafeMutableBufferPointer(start: logitsPtr, count: Int(maxCapacity))
            logitsBuf[writeIndex] = logits ? 1 : 0
        } else {
            fatalError("batch.logits is nil")
        }

        batch.n_tokens += 1
    }

    
    private func llama_batch_clear(_ batch: inout llama_batch) {
        batch.n_tokens = 0
    }
    
    /// Append bytes and try to decode a valid UTF8 prefix.
    /// Returns decoded string (possibly empty)
    private func appendBytesAndExtractString(_ bytes: [CChar]) -> String {
        // Convert [CChar] to [UInt8] (filter out zero terminator if present)
        var bytesU8 = bytes.map { UInt8(bitPattern: $0) }
        // Remove trailing zeros
        while bytesU8.last == 0 { bytesU8.removeLast() }

        // Initialize temp buffer if needed
        if _tempData == nil { _tempData = Data() }
        _tempData!.append(contentsOf: bytesU8)

        // Try to decode as UTF8
        if let s = String(data: _tempData!, encoding: .utf8) {
            _tempData = Data()
            return s
        }

        // If full decode fails, try largest valid prefix
        for cut in stride(from: _tempData!.count - 1, through: 0, by: -1) {
            if let s = String(data: _tempData!.prefix(cut), encoding: .utf8), !s.isEmpty {
                // Keep remainder for next time
                let remainder = _tempData!.suffix(from: cut)
                _tempData = Data(remainder)
                return s
            }
        }
        return ""
    }
    
    
    func safeSamplerSample(tokenIndex: Int32) -> llama_token {
        let sampled = llama_sampler_sample(sampling, context, tokenIndex)
        return sampled < 0 ? LLAMA_INVALID_TOKEN : sampled
    }
    
    private func resetBatch() {
        llama_batch_free(batch)
        batch = llama_batch_init(batchCapacity, embeddingSize, maxSeqIdsPerToken)
        guard batch.token != nil else {
            fatalError("llama_batch_init returned nil token pointer")
        }
    }
    
}
