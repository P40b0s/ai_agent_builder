use std::{fs::File, path::PathBuf};
mod deepseek;
pub use deepseek::DeepSeek;
mod qwen;
use futures::future::BoxFuture;
pub use qwen::{Qwen, QSize};
use candle_core::{backend::BackendDevice, quantized::gguf_file::{self, Content}, utils::cuda_is_available, CudaDevice, Device, Tensor};
use candle_transformers::{generation::{LogitsProcessor, Sampling}};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use crate::{token_output_stream::TokenOutputStream, ModelFromSettings, ModelName, Settings};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelSettings
{
    /// The length of the sample to generate (in tokens).
    sample_len: usize,

    /// The temperature used to generate samples, use 0 for greedy sampling.
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    top_p: Option<f64>,

    /// Only sample among the top K samples.
    top_k: Option<usize>,

    /// The seed to use when generating random samples.
    seed: u64,

    /// Process prompt elements separately.
    split_prompt: bool,

    /// Run on CPU rather than GPU even if a GPU is available.
    cpu: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    repeat_last_n: usize,
}

impl Default for ModelSettings
{
    fn default() -> Self 
    {
        ModelSettings
        {
            sample_len: 1000,
            temperature: 0.7,
            top_p: Some(0.8),
            top_k: Some(20),
            seed: 299792458,
            split_prompt: false,
            cpu: true,
            repeat_penalty: 1.5,
            repeat_last_n: 64,
        }
    }
}

pub trait AIModel
{
    fn get_name(&self) -> &'static str;
    fn get_size(&self) -> &'static str;
    ///if not exists, download file and load 
    fn tokenizer(&self) -> anyhow::Result<Tokenizer>;
    ///if not exists, download model and get path
    fn model_file_path(&self) -> anyhow::Result<PathBuf>;
    fn get_settings(&self) -> &ModelSettings;
    fn get_promt(&self, txt:&str) -> String;
    fn get_eof(&self) -> &'static str;
    fn get_device(&self) -> anyhow::Result<candle_core::Device> 
    {
        if cuda_is_available()
        {
            Ok(Device::Cuda(CudaDevice::new(0)?))
        }
        else 
        {
            Ok(Device::Cpu)
        }
    }
    fn format_size(&self, size_in_bytes: usize) -> String 
    {
        if size_in_bytes < 1_000 
        {
            format!("{size_in_bytes}B")
        } 
        else if size_in_bytes < 1_000_000 
        {
            format!("{:.2}KB", size_in_bytes as f64 / 1e3)
        } 
        else if size_in_bytes < 1_000_000_000 
        {
            format!("{:.2}MB", size_in_bytes as f64 / 1e6)
        } 
        else 
        {
            format!("{:.2}GB", size_in_bytes as f64 / 1e9)
        }
    }
    fn forward(&mut self, input: &Tensor, offset: usize) -> anyhow::Result<Tensor>;
    fn set_system_prompt(&mut self, prompt: &str);

}

pub trait GGUFContentBuilder
{
    fn load_tensors<'a>(&'a self) -> anyhow::Result<(Content, File)> where Self: AIModel
    {
        let start = std::time::Instant::now();
        let path = self.model_file_path()?;
        let mut file = std::fs::File::open(&path)?;
        let content = 
        {
            let model = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(path))?;
            let mut total_size_in_bytes = 0;
            for (_, tensor) in model.tensor_infos.iter() 
            {
                let elem_count = tensor.shape.elem_count();
                total_size_in_bytes +=
                    elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
            }
            logger::info!(
                "loaded {:?} tensors ({}) in {:.2}s",
                model.tensor_infos.len(),
                &self.format_size(total_size_in_bytes),
                start.elapsed().as_secs_f32(),
            );
            model
        };
        Ok((content, file))

    }
}



pub struct ModelWorker<T: AIModel + Send + Sync>
{
    tokenizer: Tokenizer,
    model: T
}

impl<T: AIModel + Send + Sync> ModelWorker<T>
{
    pub fn new(model: T) -> anyhow::Result<Self>
    {
        let tokenizer = model.tokenizer()?;
        Ok(Self 
        { 
            tokenizer,
            model
        })
    }
    pub fn get_token_stream(&self) -> TokenOutputStream
    {
        TokenOutputStream::new(self.tokenizer.clone())
    }
    pub fn get_logits_processor(&self) -> LogitsProcessor
    {
        let logits_processor = 
        {
            let temperature = self.model.get_settings().temperature;
            let sampling = if temperature <= 0. 
            {
                Sampling::ArgMax
            } 
            else 
            {
                match (self.model.get_settings().top_k, self.model.get_settings().top_p) 
                {
                    (None, None) => Sampling::All { temperature },
                    (Some(k), None) => Sampling::TopK { k, temperature },
                    (None, Some(p)) => Sampling::TopP { p, temperature },
                    (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
                }
            };
            LogitsProcessor::from_sampling(self.model.get_settings().seed, sampling)
        };
        logits_processor
    }
    // pub async fn prompt(&mut self, txt: &str, sender: tokio::sync::mpsc::UnboundedSender<String>) -> anyhow::Result<()>
    // {
    //     let mut tos = self.get_token_stream();
    //     let prompt_str = self.model.get_promt(txt);
    //     let tokens = tos
    //         .tokenizer()
    //         .encode(prompt_str, true)
    //         .map_err(anyhow::Error::msg)?;

    //     let tokens = tokens.get_ids();
    //     let to_sample = self.model.get_settings().sample_len.saturating_sub(1);
    //     let mut all_tokens = vec![];
    //     let mut logits_processor = self.get_logits_processor();
    //     let eos_token = *tos.tokenizer().get_vocab(true).get(self.model.get_eof()).unwrap();

    //     let start_prompt_processing = std::time::Instant::now();
    //     let mut next_token = if !self.model.get_settings().split_prompt 
    //     {
    //         let input = Tensor::new(tokens, &self.model.get_device()?)?.unsqueeze(0)?;
    //         let logits = self.model.forward(&input, 0)?;
    //         let logits = logits.squeeze(0)?;
    //         logits_processor.sample(&logits)?
    //     } 
    //     else 
    //     {
    //         let mut next_token = 0;
    //         for (pos, token) in tokens.iter().enumerate() 
    //         {
    //             let input = Tensor::new(&[*token], &self.model.get_device()?)?.unsqueeze(0)?;
    //             let logits = self.model.forward(&input, pos)?;
    //             let logits = logits.squeeze(0)?;
    //             next_token = logits_processor.sample(&logits)?
    //         }
    //         next_token
    //     };

    //     let prompt_dt = start_prompt_processing.elapsed();

    //     all_tokens.push(next_token);

       
    //     if let Some(t) = tos.next_token(next_token)? 
    //     {
    //         sender.send(t).map_err(|e| anyhow::anyhow!("Ошибка отправки токена!: {}", e))?;
    //     }

    //     let start_post_prompt = std::time::Instant::now();

    //     let mut sampled = 0;
    //     for index in 0..to_sample 
    //     {
    //         let input = Tensor::new(&[next_token],  &self.model.get_device()?)?.unsqueeze(0)?;
    //         let logits = self.model.forward(&input, tokens.len() + index)?;
    //         let logits = logits.squeeze(0)?;
    //         let logits = if self.model.get_settings().repeat_penalty == 1. 
    //         {
    //             logits
    //         } 
    //         else 
    //         {
    //             let start_at = all_tokens.len().saturating_sub(self.model.get_settings().repeat_last_n);
    //             candle_transformers::utils::apply_repeat_penalty(
    //                 &logits,
    //                 self.model.get_settings().repeat_penalty,
    //                 &all_tokens[start_at..],
    //             )?
    //         };
    //         next_token = logits_processor.sample(&logits)?;
    //         all_tokens.push(next_token);
    //         if let Some(t) = tos.next_token(next_token)? 
    //         {
    //             sender.send(t).map_err(|e| anyhow::anyhow!("Ошибка отправки токена!: {}", e))?;
    //         }
    //         sampled += 1;
    //         if next_token == eos_token 
    //         {
    //             break;
    //         };
    //     }

    //     if let Some(rest) = tos.decode_rest().map_err(candle_core::Error::msg)? 
    //     {
    //         sender.send(rest).map_err(|e| anyhow::anyhow!("Ошибка отправки оставшегося текста!: {}", e))?;
    //     }
    //     let dt = start_post_prompt.elapsed();
    //     logger::info!(
    //         "\n\n{:4} prompt tokens processed: {:.2} token/s",
    //         tokens.len(),
    //         tokens.len() as f64 / prompt_dt.as_secs_f64(),
    //     );
    //     logger::info!(
    //         "{sampled:4} tokens generated: {:.2} token/s",
    //         sampled as f64 / dt.as_secs_f64(),
    //     );
    //     Ok(())
    // }
    // pub fn prompt(&mut self, txt: &str) -> anyhow::Result<()>
    // {
    //     let mut tos = self.get_token_stream();
    //     let prompt_str = self.model.get_promt(txt);

    //     let tokens = tos
    //         .tokenizer()
    //         .encode(prompt_str, true)
    //         .map_err(anyhow::Error::msg)?;

    //     let tokens = tokens.get_ids();

    //     let to_sample = self.model.get_settings().sample_len.saturating_sub(1);

    //     let mut all_tokens = vec![];
    //     let mut logits_processor = self.get_logits_processor();

    //     let start_prompt_processing = std::time::Instant::now();

    //     let mut next_token = if !self.model.get_settings().split_prompt 
    //     {
    //         let input = Tensor::new(tokens, &self.model.get_device()?)?.unsqueeze(0)?;
    //         let logits = self.model.forward(&input, 0)?;
    //         let logits = logits.squeeze(0)?;
    //         logits_processor.sample(&logits)?
    //     } 
    //     else 
    //     {
    //         let mut next_token = 0;
    //         for (pos, token) in tokens.iter().enumerate() 
    //         {
    //             let input = Tensor::new(&[*token], &self.model.get_device()?)?.unsqueeze(0)?;
    //             let logits = self.model.forward(&input, pos)?;
    //             let logits = logits.squeeze(0)?;
    //             next_token = logits_processor.sample(&logits)?
    //         }
    //         next_token
    //     };

    //     let prompt_dt = start_prompt_processing.elapsed();

    //     all_tokens.push(next_token);

    //     if let Some(t) = tos.next_token(next_token)? 
    //     {
    //         print!("{t}");
    //         std::io::stdout().flush()?;
    //     }

    //     let eos_token = *tos.tokenizer().get_vocab(true).get(self.model.get_eof()).unwrap();

    //     let start_post_prompt = std::time::Instant::now();

    //     let mut sampled = 0;
    //     for index in 0..to_sample 
    //     {
    //         let input = Tensor::new(&[next_token],  &self.model.get_device()?)?.unsqueeze(0)?;
    //         let logits = self.model.forward(&input, tokens.len() + index)?;
    //         let logits = logits.squeeze(0)?;
    //         let logits = if self.model.get_settings().repeat_penalty == 1. 
    //         {
    //             logits
    //         } 
    //         else 
    //         {
    //             let start_at = all_tokens.len().saturating_sub(self.model.get_settings().repeat_last_n);
    //             candle_transformers::utils::apply_repeat_penalty(
    //                 &logits,
    //                 self.model.get_settings().repeat_penalty,
    //                 &all_tokens[start_at..],
    //             )?
    //         };
    //         next_token = logits_processor.sample(&logits)?;
    //         all_tokens.push(next_token);
    //         if let Some(t) = tos.next_token(next_token)? 
    //         {
    //             print!("{t}");
    //             std::io::stdout().flush()?;
    //         }
    //         sampled += 1;
    //         if next_token == eos_token 
    //         {
    //             break;
    //         };
    //     }

    //     if let Some(rest) = tos.decode_rest().map_err(candle_core::Error::msg)? 
    //     {
    //         print!("{rest}");
    //     }

    //     std::io::stdout().flush()?;
    //     let dt = start_post_prompt.elapsed();
    //     println!(
    //         "\n\n{:4} prompt tokens processed: {:.2} token/s",
    //         tokens.len(),
    //         tokens.len() as f64 / prompt_dt.as_secs_f64(),
    //     );
    //     println!(
    //         "{sampled:4} tokens generated: {:.2} token/s",
    //         sampled as f64 / dt.as_secs_f64(),
    //     );
    //     Ok(())
    // }
}
///Дает возможность передавить запрос модели
pub trait ModelPrompt 
{
    fn prompt<'a>(
        &'a mut self,
        txt: &'a str,
        sender: tokio::sync::mpsc::UnboundedSender<String>,
    ) -> BoxFuture<'a, anyhow::Result<()>>;
}

impl<T: AIModel + Send + Sync> ModelPrompt for ModelWorker<T>
{
    fn prompt<'a>(&'a mut self, txt: &'a str, sender: tokio::sync::mpsc::UnboundedSender<String>,)
     -> BoxFuture<'a, anyhow::Result<()>>
    {
        Box::pin(async move {
        let mut tos = self.get_token_stream();
        let prompt_str = self.model.get_promt(txt);
        let tokens = tos
            .tokenizer()
            .encode(prompt_str, true)
            .map_err(anyhow::Error::msg)?;

        let tokens = tokens.get_ids();
        let to_sample = self.model.get_settings().sample_len.saturating_sub(1);
        let mut all_tokens = vec![];
        let mut logits_processor = self.get_logits_processor();
        let eos_token = *tos.tokenizer().get_vocab(true).get(self.model.get_eof()).unwrap();

        let start_prompt_processing = std::time::Instant::now();
        let mut next_token = if !self.model.get_settings().split_prompt 
        {
            let input = Tensor::new(tokens, &self.model.get_device()?)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, 0)?;
            let logits = logits.squeeze(0)?;
            logits_processor.sample(&logits)?
        } 
        else 
        {
            let mut next_token = 0;
            for (pos, token) in tokens.iter().enumerate() 
            {
                let input = Tensor::new(&[*token], &self.model.get_device()?)?.unsqueeze(0)?;
                let logits = self.model.forward(&input, pos)?;
                let logits = logits.squeeze(0)?;
                next_token = logits_processor.sample(&logits)?
            }
            next_token
        };

        let prompt_dt = start_prompt_processing.elapsed();

        all_tokens.push(next_token);

       
        if let Some(t) = tos.next_token(next_token)? 
        {
            sender.send(t).map_err(|e| anyhow::anyhow!("Ошибка отправки токена!: {}", e))?;
        }

        let start_post_prompt = std::time::Instant::now();

        let mut sampled = 0;
        for index in 0..to_sample 
        {
            let input = Tensor::new(&[next_token],  &self.model.get_device()?)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, tokens.len() + index)?;
            let logits = logits.squeeze(0)?;
            let logits = if self.model.get_settings().repeat_penalty == 1. 
            {
                logits
            } 
            else 
            {
                let start_at = all_tokens.len().saturating_sub(self.model.get_settings().repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.model.get_settings().repeat_penalty,
                    &all_tokens[start_at..],
                )?
            };
            next_token = logits_processor.sample(&logits)?;
            all_tokens.push(next_token);
            if let Some(t) = tos.next_token(next_token)? 
            {
                if !sender.is_closed()
                {
                    sender.send(t).map_err(|e| anyhow::anyhow!("Ошибка отправки токена!: {}", e))?;
                }
                else
                {
                    logger::warn!("Операция отменена юзером");
                    break;
                }
            }
            sampled += 1;
            if next_token == eos_token 
            {
                break;
            };
        }
        if let Some(rest) = tos.decode_rest().map_err(candle_core::Error::msg)? 
        {
            sender.send(rest).map_err(|e| anyhow::anyhow!("Ошибка отправки оставшегося текста!: {}", e))?;
        }
        let dt = start_post_prompt.elapsed();
        logger::info!(
            "\n\n{:4} prompt tokens processed: {:.2} token/s",
            tokens.len(),
            tokens.len() as f64 / prompt_dt.as_secs_f64(),
        );
        logger::info!(
            "{sampled:4} tokens generated: {:.2} token/s",
            sampled as f64 / dt.as_secs_f64(),
        );
        Ok(())
        })
    }
}

///В зависимости от настроек получаем нужную модель и всего один метод - передача запроса для модели
pub struct PrompterBuilder
{

}
impl PrompterBuilder
{
    pub fn get_prompter() -> anyhow::Result<Box<dyn ModelPrompt>>
    {
         let settings = Settings::load()?;
        //сделал динамический диспатч чтобы не морочиться не несколькими моделями если они будт
        let prompter: Box<dyn ModelPrompt> = match settings.model_name
        {
            ModelName::Qwen => 
            {
                let model = Qwen::from_settings(settings)?;
                Box::new(ModelWorker::new(model)?)
            }
            ModelName::DeepSeek => 
            {
                let model = DeepSeek::from_settings(settings)?;
                Box::new(ModelWorker::new(model)?)
            }
        };
        Ok(prompter)
        //let prompter_mut = prompter.as_mut();
    }
}