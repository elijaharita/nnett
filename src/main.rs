use eframe::{
    egui::{panel::{Side, TopBottomSide}, CentralPanel, Context, Painter, SidePanel, TopBottomPanel},
    epaint::{pos2, vec2, Color32, Pos2, Rect, Stroke, Vec2},
    epi::{App, Frame},
    NativeOptions,
};
use itertools::Itertools;
use network::Network;

mod network;

fn main() {
    let network = Network::new(vec![28 * 28, 100, 10]);
    let network_app = NetworkApp::new(network);

    eframe::run_native(Box::new(network_app), NativeOptions::default());
}

pub struct NetworkApp {
    network: Network,
    node_size: f32,
    node_spacing: f32,
    layer_spacing: f32,
}

impl NetworkApp {
    pub fn new(network: Network) -> Self {
        Self {
            network,
            node_size: 10.0,
            node_spacing: 15.0,
            layer_spacing: 100.0,
        }
    }
}

impl App for NetworkApp {
    fn name(&self) -> &str {
        "Network Visualization"
    }

    fn update(&mut self, ctx: &Context, frame: &Frame) {
        TopBottomPanel::new(TopBottomSide::Top, "info_panel").show(ctx, |ui| {
            
        });

        CentralPanel::default().show(ctx, |ui| {
            let painter = ui.painter();

            let mut positions = Vec::new();

            let mut x = self.layer_spacing;
            for layer in self.network.layers().iter() {
                let mut layer_positions = Vec::new();

                let mut y = self.node_spacing;
                for i in 0..layer.nodes().len() {
                    if i % 28 == 0 {
                        y = self.node_spacing;
                        x += self.node_spacing;
                    }

                    layer_positions.push(pos2(x, y));

                    // Advance Y position
                    y += self.node_spacing;
                }
                positions.push(layer_positions);

                // Advance X position
                x += self.layer_spacing;
            }

            for (layer, positions) in self.network.layers().iter().zip(positions.iter()) {
                for (node, &position) in layer.nodes().iter().zip(positions.iter()) {
                    painter.rect_filled(
                        Rect::from_min_max(
                            position - Vec2::splat(self.node_size * 0.5),
                            position + Vec2::splat(self.node_size * 0.5),
                        ),
                        0.0,
                        Color32::from_rgb(255, 255, 255),
                    );
                }
            }

            for ((layer_a, positions_a), (_, positions_b)) in self
                .network
                .layers()
                .iter()
                .zip(positions.iter())
                .tuple_windows()
            {
                for (node_a, &position_a) in layer_a.nodes().iter().zip(positions_a.iter()) {
                    for &position_b in positions_b.iter() {
                        painter.line_segment(
                            [position_a, position_b],
                            Stroke::new(0.5, Color32::from_rgb(127, 127, 255)),
                        );
                    }
                }
            }
        });
    }
}
