use eframe::egui;
use egui::epaint::tessellator::Path;
use egui::{Color32, ColorImage};
use image::ColorType::Rgb32F;
use image::{DynamicImage, GrayImage, Luma, imageops};
use image::{GenericImageView, io::Reader as ImageReader};
use image::{ImageBuffer, ImageFormat};
use image::{Rgb, RgbImage};
use imageproc::contrast::adaptive_threshold;
use imageproc::distance_transform::Norm;
use imageproc::morphology::{dilate, erode};
use opencv::photo;
use opencv::prelude::*;
use opencv::{Result, core, imgcodecs, imgproc, prelude::*, types};
use opencv::{core::Mat, prelude::*};
use opencv::{
    core::{Point, Scalar, Vector},
    prelude::*,
};

use std::default;
use std::fs;
use std::io::Cursor;
use std::io::Read;
use std::sync::mpsc;

mod denoiser_app;
mod denoiser_params;
mod ocr_wrapper;
mod utils;

use denoiser_app::DenoiserApp;
use denoiser_params::DenoiserParams;
use utils::mat_to_color_image;
use utils::mat_to_png;

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

                // ======= Density ========
                ui.label("Density map");
                if let Ok(bytes) = self.rx_density.try_recv() {
                    Self::load_texture_from_bytes(ui, bytes, &mut self.texture_denity);
                }

                let size = egui::vec2(400.0, 100.0);

                let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());

                ui.painter().rect_filled(rect, 0.0, egui::Color32::BLACK);

                if let Some(texture) = &self.texture_denity {
                    ui.put(rect, egui::Image::new(texture).fit_to_exact_size(size));
                } else {
                    ui.put(
                        rect,
                        egui::Label::new("No image").sense(egui::Sense::hover()),
                    );
                }

                // ======= Lines ========
                ui.label("Lines detection");

                let size = egui::vec2(400.0, 100.0);

                let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());

                ui.painter().rect_filled(rect, 0.0, egui::Color32::BLACK);

                if let Ok(color_image) = self.rx_lines.try_recv() {
                    Self::load_texture_from_color_image(ui, color_image, &mut self.texture_lines);
                }

                if let Some(texture) = &self.texture_lines {
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

                let mut changed = false;
                ui.add_space(20.0);
                ui.vertical_centered(|ui| {
                    // 1. Constrain the width so the "center" is meaningful
                    ui.set_max_width(400.0);
                    egui::Grid::new("my_grid")
                        .num_columns(2)
                        .spacing([40.0, 4.0]) // [horizontal, vertical] spacing
                        .show(ui, |ui| {
                            ui.add(egui::Label::new("Block"));
                            if ui
                                .add(
                                    egui::DragValue::new(&mut self.params.block)
                                        .range(0.0..=99999.0),
                                )
                                .changed()
                            {
                                self.denoise();
                            };

                            ui.add(egui::Label::new("RHO"));
                            if ui
                                .add(
                                    egui::DragValue::new(&mut self.params.rho).range(0.0..=99999.0),
                                )
                                .changed()
                            {
                                self.denoise();
                            };

                            ui.end_row();

                            ui.add(egui::Label::new("Delta"));
                            if ui
                                .add(
                                    egui::DragValue::new(&mut self.params.delta)
                                        .range(0.0..=99999.0),
                                )
                                .changed()
                            {
                                self.denoise();
                            };

                            ui.add(egui::Label::new("Theta"));
                            if ui
                                .add(
                                    egui::DragValue::new(&mut self.params.theta)
                                        .speed(0.001)
                                        .range(0.01..=99999.0),
                                )
                                .changed()
                            {
                                self.denoise();
                            };

                            ui.end_row();

                            ui.add(egui::Label::new("Erase K"));
                            if ui
                                .add(
                                    egui::DragValue::new(&mut self.params.erase_k)
                                        .range(0.0..=99999.0),
                                )
                                .changed()
                            {
                                self.denoise();
                            };

                            ui.add(egui::Label::new("Line_threshold"));
                            if ui
                                .add(
                                    egui::DragValue::new(&mut self.params.line_threshold)
                                        .range(0.0..=99999.0),
                                )
                                .changed()
                            {
                                self.denoise();
                            };

                            ui.end_row();

                            ui.add(egui::Label::new("Dilate K"));
                            if ui
                                .add(
                                    egui::DragValue::new(&mut self.params.dilate_k)
                                        .range(0.0..=99999.0),
                                )
                                .changed()
                            {
                                self.denoise();
                            };

                            ui.add(egui::Label::new("min_line_length"));
                            if ui
                                .add(
                                    egui::DragValue::new(&mut self.params.min_line_length)
                                        .range(0.0..=99999.0),
                                )
                                .changed()
                            {
                                self.denoise();
                            };

                            ui.end_row();

                            ui.add(egui::Label::new("Threshold"));
                            if ui
                                .add(
                                    egui::DragValue::new(&mut self.params.threshold)
                                        .range(0.0..=99999.0),
                                )
                                .changed()
                            {
                                self.denoise();
                            };

                            ui.add(egui::Label::new("min_line_gap"));
                            if ui
                                .add(
                                    egui::DragValue::new(&mut self.params.min_line_gap)
                                        .range(0.0..=99999.0),
                                )
                                .changed()
                            {
                                self.denoise();
                            };

                            ui.end_row();
                            //=======================

                            ui.add(egui::Label::new("Low Threshold"));
                            if ui
                                .add(
                                    egui::DragValue::new(&mut self.params.low_threshold)
                                        .range(0.0..=99999.0),
                                )
                                .changed()
                            {
                                self.denoise();
                            };

                            ui.add(egui::Label::new("High threshold"));
                            if ui
                                .add(
                                    egui::DragValue::new(&mut self.params.high_threshold)
                                        .range(0.0..=99999.0),
                                )
                                .changed()
                            {
                                self.denoise();
                            };

                            ui.end_row();

                            ui.add(egui::Label::new("Aperture"));
                            if ui
                                .add(egui::DragValue::new(&mut self.params.aperture).range(3..=7))
                                .changed()
                            {
                                self.denoise();
                            };

                            ui.add(egui::Label::new("l2_gradient"));
                            if ui
                                .checkbox(&mut self.params.l2_gradient, "l2_gradient")
                                .changed()
                            {
                                self.denoise();
                            };

                            ui.end_row();

                            ui.add(egui::Label::new("inpaint_radius"));
                            if ui
                                .add(
                                    egui::DragValue::new(&mut self.params.inpaint_radius)
                                        .range(0.0..=99999.0)
                                        .speed(0.1),
                                )
                                .changed()
                            {
                                self.denoise();
                            };

                            ui.end_row();

                            ui.add(egui::Label::new("Horizontal line thickness"));
                            if ui
                                .add(
                                    egui::DragValue::new(&mut self.params.hl_thickness)
                                        .range(1..=10)
                                        .speed(1),
                                )
                                .changed()
                            {
                                self.denoise();
                            };

                            ui.add(egui::Label::new("Horizontal line step"));
                            if ui
                                .add(
                                    egui::DragValue::new(&mut self.params.hl_step)
                                        .range(1..=10)
                                        .speed(1),
                                )
                                .changed()
                            {
                                self.denoise();
                            };

                            ui.end_row();

                            ui.add(egui::Label::new("Vertical line thickness"));
                            if ui
                                .add(
                                    egui::DragValue::new(&mut self.params.vl_thickness)
                                        .range(1..=10)
                                        .speed(1),
                                )
                                .changed()
                            {
                                self.denoise();
                            };

                            ui.add(egui::Label::new("Vertical line step"));
                            if ui
                                .add(
                                    egui::DragValue::new(&mut self.params.vl_step)
                                        .range(1..=10)
                                        .speed(1),
                                )
                                .changed()
                            {
                                self.denoise();
                            };

                            ui.end_row();
                        });
                });

                ui.add_space(20.0);
                // Buttons
                let btn = egui::Button::new(egui::RichText::new("Denoise").size(20.0));
                ui.vertical_centered(|ui| {
                    if ui.add(btn).clicked() {
                        self.denoise();
                    }
                });
                ui.add_space(10.0);
                let btn = egui::Button::new(egui::RichText::new("OCR").size(20.0));
                ui.vertical_centered(|ui| {
                    if ui.add(btn).clicked() {
                        self.ocr();
                    }
                });

                if let Ok(lock) = self.ocr_worker.text.lock() {
                    if let Some(text) = lock.clone() {
                        ui.label(text);
                    }
                }
            });
        });
    }
}

#[tokio::main]
async fn main() {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size(egui::vec2(540.0, 900.0)),
        ..Default::default()
    };

    let path = r"captcha.png".to_string();

    let _ = eframe::run_native(
        "Denoiser",
        options,
        Box::new(|cc| Ok(Box::new(DenoiserApp::new(cc, path)))),
    );
}
