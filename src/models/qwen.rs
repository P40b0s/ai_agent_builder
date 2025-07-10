use std::{path::Path};
use tokenizers::Tokenizer;

use crate::{models::{AIModel, GGUFContentBuilder, ModelSettings}, ModelSize, ModelFromSettings};

pub enum QSize
{
    W0_6B,
    W1_7B,
    W4B,
    W7B
}
impl Into<QSize> for ModelSize
{
    fn into(self) -> QSize 
    {
        match self
        {
            ModelSize::W0_6B => QSize::W0_6B,
            ModelSize::W1_7B => QSize::W1_7B,
            ModelSize::W4B => QSize::W4B,
            ModelSize::W7B => QSize::W7B
        }
    }
}

pub struct Qwen
{
    settings: ModelSettings,
    size: QSize,
    system_prompt: Option<String>,
    model: Option<candle_transformers::models::quantized_qwen3::ModelWeights>
}
impl Default for Qwen
{
    fn default() -> Self 
    {
        Self { settings: ModelSettings::default(), model: None , size: QSize::W0_6B, system_prompt: None}
    }
}

impl Qwen
{
    // pub fn new_with_settings(settings: ModelSettings, size: QSize) -> anyhow::Result<Self> 
    // {
    //     let slf = Self::default();
    //     let model = slf.get_model()?;
    //     Ok(Self 
    //     { 
    //         settings,
    //         model: Some(model),
    //         size,
    //         system_prompt: None
    //     })
    // }
    // pub fn new(size: QSize) -> anyhow::Result<Self> 
    // {
    //     let slf = Self::default();
    //     let model = slf.get_model()?;
    //     Ok(Self 
    //     { 
    //         settings: slf.settings,
    //         model: Some(model),
    //         size,
    //         system_prompt: None
    //     })
    // }
    
    fn get_model(&self) -> anyhow::Result<candle_transformers::models::quantized_qwen3::ModelWeights> 
    {
        let (content, mut file) = self.load_tensors()?;
        let device = self.get_device()?;
        let model = candle_transformers::models::quantized_qwen3::ModelWeights::from_gguf(content,  &mut file, &device)?;
        logger::info!("model built");
        Ok(model)
    }
}

impl GGUFContentBuilder for Qwen{}

impl super::AIModel for Qwen
{
    fn get_name(&self) -> &'static str 
    {
        match self.size
        {
            QSize::W0_6B => "Qwen3-0.6B-Q4_K_M.gguf",
            QSize::W1_7B => "Qwen3-1.7B-Q8_0.gguf",
            QSize::W4B => "Qwen3-4B-Q6_K.gguf",
            QSize::W7B => "Qwen3-7B-Q6_K.gguf"
        }
    }

    fn get_size(&self) -> &'static str 
    {
        match self.size
        {
            QSize::W0_6B => "0.6B",
            QSize::W1_7B => "1.7B",
            QSize::W4B => "4B",
            QSize::W7B => "7B"
        }
    }
    fn set_system_prompt(&mut self, prompt: &str) 
    {
        self.system_prompt = Some(prompt.to_owned());
    }

    fn tokenizer(&self) -> anyhow::Result<tokenizers::Tokenizer> 
    {
        let path = Path::new("models").join("qwen").join("tokenizer.json");
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
         let path = Path::new("models").join("qwen").join(self.get_name());
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
        if let Some(sp) = self.system_prompt.as_ref()
        {
            let prompt = format!("<|im_start|>system {sp}<|im_end|>
            <|im_start|>user\n{prompt_str}<|im_end|>\n<|im_start|>assistant\n");
            prompt
        }
        else 
        {
            logger::warn!("Системный промт не задан, агент не имеет специализации!");
            let prompt = format!("<|im_start|>user\n{prompt_str}<|im_end|>\n<|im_start|>assistant\n");
            prompt
        }
    }
    fn get_eof(&self) -> &'static str
    {
        "<|im_end|>"
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

impl ModelFromSettings for Qwen
{
    fn from_settings(settings: crate::Settings) -> anyhow::Result<Self>
    {
        let size: QSize = settings.model_size.into();
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
            size,
            system_prompt: Some(settings.system_prompt)
        })
    }
}