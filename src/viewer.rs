pub struct Viewer {
    events: std::sync::mpsc::Receiver<(Option<String>, Option<u32>, Option<String>)>,
    best_match_path: Option<String>,
    score: Option<u32>,
    set_symbol: Option<String>,
}

#[derive(Debug)]
pub enum ViewerMessage {
    Poll,
}

impl iced::Application for Viewer {
    type Executor = iced::executor::Default;
    type Flags = std::sync::mpsc::Receiver<(Option<String>, Option<u32>, Option<String>)>;
    type Message = ViewerMessage;

    fn new(flags: Self::Flags) -> (Self, iced::Command<Self::Message>) {
        (Viewer {
            events: flags,
            best_match_path: None,
            score: None,
            set_symbol: None,
        }, iced::Command::none())
    }

    fn title(&self) -> String {
        "Viewer".to_string()
    }

    fn update(&mut self, message: Self::Message, _debug: &mut iced::Clipboard) -> iced::Command<Self::Message> {
        match message {
            ViewerMessage::Poll => {
                println!("polling");
                match self.events.try_recv() {
                    Ok(event) => {
                        self.best_match_path = event.0;
                        self.score = event.1;
                        self.set_symbol = event.2;
                        println!("match: {:?}", self.best_match_path);
                        println!("score: {:?}", self.score);
                        println!("set: {:?}", self.set_symbol);
                    },
                    _ => {}
                }
            }
        }

        iced::Command::none()
    }

    fn subscription(&self) -> iced::Subscription<Self::Message> {
        iced_futures::time::every(std::time::Duration::from_millis(500)).map(|_| ViewerMessage::Poll)
    }

    fn view(&mut self) -> iced::Element<Self::Message> {
        iced::Column::new()
            .padding(20)
            .spacing(10)
            .width(iced::Length::Fill)
            .height(iced::Length::Fill)
            .push(
                iced::image::Image::new(self.best_match_path.clone().unwrap_or("cardback.jpg".to_string()))
                .height(iced::Length::Units(512))
            )
            .push(
                iced::Row::new()
                .width(iced::Length::Fill)
                .push(iced::Text::new("Score").width(iced::Length::Fill))
                .push(iced::Text::new(self.score.map(|x| format!("{}", x)).unwrap_or("--".to_string()))),
            )
            .push(
                iced::Row::new()
                .width(iced::Length::Fill)
                .push(iced::Text::new("Set").width(iced::Length::Fill))
                .push(iced::Svg::from_path(format!("templates/{}.svg", self.set_symbol.clone().unwrap_or("--".to_string()))).height(iced::Length::Units(40)))
            )
            .into()
    }
}


