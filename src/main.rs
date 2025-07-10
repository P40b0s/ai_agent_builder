mod candle_test;
mod settings;
pub use settings::{Settings, ModelFromSettings, ModelName, ModelSize};
use std::io::{self, Write};
use crate::models::{DeepSeek, ModelPrompt, ModelWorker, PrompterBuilder, QSize, Qwen};
mod token_output_stream;
mod models;
#[tokio::main]
async fn main() -> anyhow::Result<()>
{
    let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
    //TODO Дропаем сендер и получаем отмену опреции генерации токенов
    //для командной строки в этом нет особого смысла так как при нажатии ctrl+c выключается вся программа
    let _ = logger::StructLogger::new_default();
    tokio::spawn(async move 
    {
        while let Some(token) = receiver.recv().await 
        {
            print!("{}", token);
            std::io::stdout().flush().unwrap();
        }
    });
    
    let mut prompter = PrompterBuilder::get_prompter()?;
    let prompter_mut = prompter.as_mut();

    loop 
    {
        let sender = sender.clone();
        print!(">");  // Приглашение для ввода
        io::stdout().flush().unwrap();  // Сброс буфера вывода
        
        let mut input = String::new();
        
        // Чтение строки ввода
        match io::stdin().read_line(&mut input) 
        {
            Ok(0) => 
            {
                // Обнаружен EOF (Ctrl+D в Unix/Linux, Ctrl+Z+Enter в Windows)
                println!("\nЗавершение ввода (EOF)");
                break;
            }
            Ok(_) => 
            {
                // Удаляем символы перевода строки
                let input = input.trim_end().to_string();
                
                // Проверка на команду выхода
                if input.eq_ignore_ascii_case("exit") 
                {
                    println!("Завершение ввода по команде");
                    break;
                }
                
                // Добавляем непустые строки
                if !input.is_empty() 
                {
                    prompter_mut.prompt(&input, sender).await?;
                    //worker.prompt_signal(&input, sender).await?;
                }
            }
            Err(e) => 
            {
                return Err(anyhow::Error::msg(["Ошибка чтения ввода: ", &e.to_string()].concat()))
            }
        }
    }
    Ok(())
}

// fn main()
// {
//     //let mut model = candle_test::OrfoModel::load_model().unwrap();
//     let model = Qwen::new(QSize::W0_6B).unwrap();
//     let mut worker = ModelWorker::new(model).unwrap();
//     loop 
//     {
//         print!(">");  // Приглашение для ввода
//         io::stdout().flush().unwrap();  // Сброс буфера вывода
        
//         let mut input = String::new();
        
//         // Чтение строки ввода
//         match io::stdin().read_line(&mut input) 
//         {
//             Ok(0) => 
//             {
//                 // Обнаружен EOF (Ctrl+D в Unix/Linux, Ctrl+Z+Enter в Windows)
//                 println!("\nЗавершение ввода (EOF)");
//                 break;
//             }
//             Ok(_) => 
//             {
//                 // Удаляем символы перевода строки
//                 let input = input.trim_end().to_string();
                
//                 // Проверка на команду выхода
//                 if input.eq_ignore_ascii_case("exit") 
//                 {
//                     println!("Завершение ввода по команде");
//                     break;
//                 }
                
//                 // Добавляем непустые строки
//                 if !input.is_empty() 
//                 {
//                     worker.prompt(&input);
//                 }
//             }
//             Err(e) => 
//             {
//                 eprintln!("Ошибка чтения ввода: {}", e);
//                 break;
//             }
//         }
//     }
// }