use eframe::egui;
use egui::epaint::tessellator::Path;
use image::{DynamicImage, GrayImage, Luma, imageops};
use imageproc::contrast::adaptive_threshold;
use imageproc::distance_transform::Norm;
use imageproc::morphology::{dilate, erode};
use std::io::Read;
use std::sync::mpsc;
use std::{default, fs};

mod denoiser_params;

use denoiser_params::DenoiserParams;

struct DenoiserApp {
    tx_orig: mpsc::Sender<Vec<u8>>,
    rx_orig: mpsc::Receiver<Vec<u8>>,
    tx_denoised: mpsc::Sender<Vec<u8>>,
    rx_denoised: mpsc::Receiver<Vec<u8>>,
    tx_text: mpsc::Sender<Vec<u8>>,
    rx_text: mpsc::Receiver<Vec<u8>>,
    texture_orig: Option<egui::TextureHandle>,
    texture_denoised: Option<egui::TextureHandle>,
    params: DenoiserParams,
}

impl Default for DenoiserApp {
    fn default() -> Self {
        let (tx_orig, rx_orig) = mpsc::channel();
        let (tx_denoised, rx_denoised) = mpsc::channel();
        let (tx_text, rx_text) = mpsc::channel();

        Self {
            tx_orig,
            rx_orig,
            tx_denoised,
            rx_denoised,
            tx_text,
            rx_text,
            texture_orig: None,
            texture_denoised: None,
            params: DenoiserParams::default(),
        }
    }
}

impl DenoiserApp {
    fn new(_cc: &eframe::CreationContext, file_path: String) -> Self {
        Self {
            params: DenoiserParams {
                path: file_path,
                ..DenoiserParams::default()
            },
            ..DenoiserApp::default()
        }
    }

    fn clean_captcha(
        raw_bytes: &[u8],
        block: u32,
        delta: i32,
        erase_k: u8,
        dilate_k: u8,
    ) -> Vec<u8> {
        println!(
            "Starting denoiser -> Block: {}; delta: {}; erase_k: {}; dilate_k: {}; ",
            block, delta, erase_k, dilate_k
        );

        // 1. Load the image and convert to Grayscale
        let img = image::load_from_memory(raw_bytes)
            .expect("Failed to load image")
            .to_luma8();

        // 2. Scale up: Captchas are usually too small for Tesseract.
        // Making it 2x or 3x larger helps OCR accuracy significantly.
        let (w, h) = img.dimensions();
        let img = imageops::resize(&img, w * 2, h * 2, imageops::FilterType::Lanczos3);

        // 3. Adaptive Thresholding: This is better than a fixed threshold for removing
        // background lines that have different color intensities.
        // '8' is the block radius; adjust this based on line thickness.
        let binarized = adaptive_threshold(&img, block, delta);

        // 4. Denoise: Use Erosion then Dilation (Opening) to remove small dots/thin lines
        let denoised = erode(&binarized, Norm::LInf, erase_k);
        let final_img = dilate(&denoised, Norm::LInf, dilate_k);

        // 5. Convert back to bytes for Tesseract
        let mut buffer = std::io::Cursor::new(Vec::new());
        final_img
            .write_to(&mut buffer, image::ImageFormat::Png)
            .expect("Failed to write to buffer");

        println!("Denoiser complete.");

        buffer.into_inner()
    }

    fn denoise(&mut self) {
        let tx_orig = self.tx_orig.clone();
        let tx_denoised = self.tx_denoised.clone();
        let params = self.params.clone();

        println!("Loading {}", params.path);

        tokio::spawn(async move {
            let bytes = fs::read(params.path).expect("Unable to read file");

            tx_orig.send(bytes.clone()).unwrap();

            // Denoising
            println!("Denoising...");
            let denoised_bytes = Self::clean_captcha(
                &bytes.clone(),
                params.block,
                params.delta,
                params.erase_k,
                params.dilate_k,
            );
            tx_denoised.send(denoised_bytes.clone()).unwrap();

            println!("Done");
        });
    }

    fn load_texture_from_bytes(
        ui: &mut egui::Ui,
        bytes: Vec<u8>,
        texture: &mut Option<egui::TextureHandle>,
    ) {
        let image = image::load_from_memory(&bytes).unwrap().to_rgba8();

        let size = [image.width() as usize, image.height() as usize];

        let color_image = egui::ColorImage::from_rgba_unmultiplied(size, image.as_raw());

        *texture = Some(ui.load_texture("captcha_original", color_image, Default::default()));
    }
}

impl eframe::App for DenoiserApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.vertical_centered(|ui| {
                ui.heading("Denoiser");

                ui.add_space(10.0);

                // ======= Original ========
                ui.label("Original");
                if let Ok(bytes) = self.rx_orig.try_recv() {
                    Self::load_texture_from_bytes(ui, bytes, &mut self.texture_orig);
                }

                let size = egui::vec2(400.0, 100.0);

                let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());

                ui.painter().rect_filled(rect, 0.0, egui::Color32::BLACK);

                if let Some(texture) = &self.texture_orig {
                    ui.put(rect, egui::Image::new(texture).fit_to_exact_size(size));
                } else {
                    ui.put(
                        rect,
                        egui::Label::new("No image").sense(egui::Sense::hover()),
                    );
                }

                // ======= Denoised ========
                ui.label("Denoised");
                if let Ok(bytes) = self.rx_denoised.try_recv() {
                    Self::load_texture_from_bytes(ui, bytes, &mut self.texture_denoised);
                }
                // Image
                let size = egui::vec2(400.0, 100.0);

                let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());

                ui.painter().rect_filled(rect, 0.0, egui::Color32::BLACK);

                if let Some(texture) = &self.texture_denoised {
                    ui.put(rect, egui::Image::new(texture).fit_to_exact_size(size));
                } else {
                    ui.put(
                        rect,
                        egui::Label::new("No image").sense(egui::Sense::hover()),
                    );
                }

                ui.add_space(20.0);

                ui.add(egui::Label::new("Block"));
                ui.add(egui::DragValue::new(&mut self.params.block));

                ui.add(egui::Label::new("Delta"));
                ui.add(egui::DragValue::new(&mut self.params.delta));

                ui.add(egui::Label::new("Erase K"));
                ui.add(egui::DragValue::new(&mut self.params.erase_k));

                ui.add(egui::Label::new("Dilate K"));
                ui.add(egui::DragValue::new(&mut self.params.dilate_k));

                ui.add_space(20.0);
                // Button
                let btn = egui::Button::new(egui::RichText::new("Denoise").size(20.0));
                ui.vertical_centered(|ui| {
                    if ui.add(btn).clicked() {
                        self.denoise();
                    }
                });
            });
        });
    }
}

#[tokio::main]
async fn main() {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size(egui::vec2(540.0, 540.0)),
        ..Default::default()
    };

    let path = r"D:\Main3\captcha.png".to_string();

    eframe::run_native(
        "Denoiser",
        options,
        Box::new(|cc| Ok(Box::new(DenoiserApp::new(cc, path)))),
    );
}
