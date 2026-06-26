use image::load_from_memory;
use pure_onnx_ocr::{OcrEngine, OcrEngineBuilder, OcrResult};

use std::sync::mpsc;
use std::thread;
pub struct OcrJob {
    pub bytes: Vec<u8>,
}

struct OcrOutput {
    text: String,
}

pub struct OcrWrapper {
    pub engine: OcrEngine,
}

impl Default for OcrWrapper {
    fn default() -> Self {
        let engine = OcrEngineBuilder::new()
            .det_model_path("src/models/det.onnx")
            .rec_model_path("src/models/rec.onnx")
            .dictionary_path("src/models/dict2.txt")
            .det_limit_side_len(960)
            .det_unclip_ratio(1.5)
            .rec_batch_size(8)
            .build()
            .unwrap();

        Self { engine: engine }
    }
}

impl OcrWrapper {
    pub fn ocr(&self, bytes: &[u8]) -> String {
        let img = load_from_memory(&bytes).unwrap();

        let results: Vec<OcrResult> = self.engine.run_from_image(&img).unwrap();
        for (idx, result) in results.iter().enumerate() {
            println!(
                "#{} text={} confidence={:.4}",
                idx, result.text, result.confidence,
            );
        }

        results
            .into_iter()
            .map(|r| r.text)
            .collect::<Vec<_>>()
            .join("\n")
    }
}

pub struct OcrWorker {
    pub tx: mpsc::Sender<OcrJob>,
    rx_result: mpsc::Receiver<OcrOutput>,
}

impl OcrWorker {
    pub fn new() -> Self {
        let (tx_job, rx_job) = mpsc::channel::<OcrJob>();
        let (tx_result, rx_result) = mpsc::channel::<OcrOutput>();

        thread::spawn(move || {
            let ocr = OcrWrapper::default();

            while let Ok(job) = rx_job.recv() {
                let text = ocr.ocr(&job.bytes);
                let _ = tx_result.send(OcrOutput { text });
            }
        });

        Self {
            tx: tx_job,
            rx_result,
        }
    }

    pub fn submit(&self, bytes: Vec<u8>) {
        let _ = self.tx.send(OcrJob { bytes });
    }

    pub fn try_recv(&self) -> Option<String> {
        self.rx_result.try_recv().ok().map(|r| r.text)
    }
}
