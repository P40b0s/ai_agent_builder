use std::path::Path;
use tokenizers::Tokenizer;

use crate::{models::{AIModel, GGUFContentBuilder, ModelSettings}, ModelFromSettings};

pub struct DeepSeek
{
    settings: ModelSettings,
    system_prompt: Option<String>,
    model: Option<candle_transformers::models::quantized_qwen2::ModelWeights>
}
impl Default for DeepSeek
{
    fn default() -> Self 
    {
        Self { settings: ModelSettings::default(), model: None, system_prompt: None }
    }
}

impl DeepSeek
{
    // pub fn new_with_settings(settings: ModelSettings) -> anyhow::Result<Self> 
    // {
    //     let slf = Self::default();
    //     let model = slf.get_model()?;
    //     Ok(Self 
    //     { 
    //         settings,
    //         model: Some(model),
    //         system_prompt: None
    //     })
    // }
    // pub fn new() -> anyhow::Result<Self> 
    // {
    //     let slf = Self::default();
    //     let model = slf.get_model()?;
    //     Ok(Self 
    //     { 
    //         settings: slf.settings,
    //         model: Some(model),
    //         system_prompt: None
    //     })
    // }
    fn get_model(&self) -> anyhow::Result<candle_transformers::models::quantized_qwen2::ModelWeights> 
    {
        let (content, mut file) = self.load_tensors()?;
        let device = self.get_device()?;
        let model = candle_transformers::models::quantized_qwen2::ModelWeights::from_gguf(content,  &mut file, &device)?;
        logger::info!("model built");
        Ok(model)
    }
}

impl GGUFContentBuilder for DeepSeek{}

impl super::AIModel for DeepSeek
{
    fn get_name(&self) -> &'static str 
    {
        "DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf"
        //let path = Path::new("models").join("deepseek").join("DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf")
    }

    fn get_size(&self) -> &'static str 
    {
        "1.5B"
    }
    fn set_system_prompt(&mut self, prompt: &str) 
    {
        self.system_prompt = Some(prompt.to_owned());
    }

    fn tokenizer(&self) -> anyhow::Result<tokenizers::Tokenizer> 
    {
        let path = Path::new("models").join("deepseek").join("tokenizer.json");
        if path.exists()
        {
            Tokenizer::from_file(path).map_err(anyhow::Error::msg)
        }
        else 
        {
            Err(anyhow::Error::msg(["Файл токенов не найден ->", &path.display().to_string()].concat()))
        }
    }

    fn model_file_path(&self) -> anyhow::Result<std::path::PathBuf> 
    {
        let path = Path::new("models").join("deepseek").join(self.get_name());
        if path.exists()
        {
            Ok(path)
        }
        else 
        {
            Err(anyhow::Error::msg(["Файл модели не найден ->", &path.display().to_string()].concat()))
        }
    }

    fn get_settings(&self) -> &super::ModelSettings 
    {
        &self.settings
    }

    fn get_promt(&self, prompt_str :&str) -> String 
    {
        //у дистиллированных моделей не нашел возможности передать системный промт
        format!("<｜User｜> {prompt_str}<｜Assistant｜>")
    }
    
    fn get_eof(&self) -> &'static str
    {
        "<｜end▁of▁sentence｜>"
    }
    fn forward(&mut self, input: &candle_core::Tensor, offset: usize) -> anyhow::Result<candle_core::Tensor> 
    {
        if let Some(model) = self.model.as_mut()
        {
            let forw = model.forward(input, offset)?;
            Ok(forw)
        }
        else 
        {
            Err(anyhow::Error::msg("Ошибка, модель не загружена!"))
        }
    }
}


impl ModelFromSettings for DeepSeek
{
    fn from_settings(settings: crate::Settings) -> anyhow::Result<Self>
    {
        let model_settings = if let Some(ms) = settings.model_settings
        {
            ms
        }
        else
        {
            ModelSettings::default()
        };
        let slf = Self::default();
        let model = slf.get_model()?;
        Ok(Self 
        { 
            settings: model_settings,
            model: Some(model),
            system_prompt: None
        })
    }
}