use std::fs::File;

use eframe::{
    egui::{panel::Side, Button, CentralPanel, Context, Image, Sense, SidePanel, Label},
    epaint::{vec2, AlphaImage, ColorImage, ImageData, TextureHandle, TextureId},
    epi::{App, Frame, Storage},
    NativeOptions,
};
use image::{load_from_memory, GrayImage};
use nalgebra as na;
use network::{ConvolutionalLayer, Network, ReluLayer, SoftMaxLayer, FullyConnectedLayer};

mod mnist;
mod network;

fn main() {
    let mut network = Network::new(na::Vector2::new(28, 28));
    network.add_layer(Box::new(ConvolutionalLayer::new(na::Vector2::new(5, 5))));
    network.add_layer(Box::new(ReluLayer::new()));
    network.add_layer(Box::new(ConvolutionalLayer::new(na::Vector2::new(5, 5))));
    network.add_layer(Box::new(ReluLayer::new()));
    network.add_layer(Box::new(FullyConnectedLayer::new(na::Vector2::new(100, 1))));
    network.add_layer(Box::new(ReluLayer::new()));
    network.add_layer(Box::new(FullyConnectedLayer::new(na::Vector2::new(10, 1))));
    network.add_layer(Box::new(SoftMaxLayer::new()));

    let images = mnist::parse_mnist_images(
        &mut File::open("assets/samples/mnist/train-images-idx3-ubyte").unwrap(),
    );
    let labels = mnist::parse_mnist_labels(
        &mut File::open("assets/samples/mnist/train-labels-idx1-ubyte").unwrap(),
    );
    let network_app = NetworkApp::new(network, images, labels);

    eframe::run_native(Box::new(network_app), NativeOptions::default());
}

pub struct NetworkApp {
    network: Network,

    node_size: f32,
    node_spacing: f32,
    layer_spacing: f32,

    images: Vec<GrayImage>,
    labels: Vec<u8>,

    curr_image_index: usize,

    input_texture: Option<TextureHandle>,
    layer_textures: Vec<TextureHandle>,
}

impl NetworkApp {
    pub fn new(network: Network, images: Vec<GrayImage>, labels: Vec<u8>) -> Self {
        Self {
            network,

            node_size: 10.0,
            node_spacing: 15.0,
            layer_spacing: 100.0,

            images,
            labels,

            curr_image_index: 0,

            input_texture: None,
            layer_textures: Vec::new(),
        }
    }
}

impl App for NetworkApp {
    fn name(&self) -> &str {
        "Network Visualization"
    }

    fn update(&mut self, ctx: &Context, frame: &Frame) {
        let curr_image = &self.images[self.curr_image_index];
        let curr_label = self.labels[self.curr_image_index];
        self.input_texture = Some(load_image_to_texture(ctx, curr_image));

        self.layer_textures = Vec::new();
        
        let output = self.network.evaluate_and(
            &curr_image
                .iter()
                .cloned()
                .map(|pixel| (pixel as f32) / 255.0)
                .collect::<Vec<_>>(),
            |layer, output| {
                let output_size = layer.output_size();
                self.layer_textures.push(load_image_to_texture(
                    ctx,
                    &GrayImage::from_vec(
                        output_size.x.try_into().unwrap(),
                        output_size.y.try_into().unwrap(),
                        output
                            .iter()
                            .map(|pixel| (pixel * 255.0) as u8)
                            .collect::<Vec<_>>(),
                    )
                    .unwrap(),
                ))
            }
        );

        // Calculate expected output (1.0 at the index of the expected number)
        let mut expected_output = vec![0.0; output.len()];
        expected_output[curr_label as usize] = 1.0;

        // Calculate error: sum of the squares of all the differences between
        // output and expected output
        let cost: f32 = output.iter().zip(expected_output).map(|(o, eo)| (o - eo) * (o - eo)).sum();

        let output_size = self.network.output_size();

        CentralPanel::default().show(ctx, |ui| {
            if ui
                .add(Button::new("next image").sense(Sense::click()))
                .clicked()
            {
                self.curr_image_index += 1;
            }

            ui.add(Label::new(format!("cost: {}", cost)));

            if let Some(input_texture) = &self.input_texture {
                ui.add(Image::new(input_texture, vec2(56.0, 56.0)));
            }

            for layer_texture in self.layer_textures.iter() {
                ui.add(Image::new(layer_texture, vec2(56.0, 56.0)));
            }
        });
    }
}

fn load_image_to_texture(ctx: &Context, image: &GrayImage) -> TextureHandle {
    ctx.load_texture(
        "input",
        ColorImage::from_rgba_unmultiplied(
            [
                image.width().try_into().unwrap(),
                image.height().try_into().unwrap(),
            ],
            &image
                .iter()
                .copied()
                .flat_map(|pixel| [255, 255, 255, pixel])
                .collect::<Vec<_>>(),
        ),
    )
}
