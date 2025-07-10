use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize};

use crate::models::ModelSettings;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelName
{
    Qwen,
    DeepSeek
}
impl FromStr for ModelName
{
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> 
    {
        match s.to_lowercase().as_str()
        {
            "qwen" => Ok(ModelName::Qwen),
            "deepseek" => Ok(ModelName::DeepSeek),
            _ => Err("Значение параметра веса модели неверно! Возможны только значения qwen, deepseek".to_owned())
        }
    }
}
impl Serialize for ModelName 
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self 
        {
            ModelName::Qwen => serializer.serialize_str("qwen"),
            ModelName::DeepSeek => serializer.serialize_str("deepseek")
        }
    }
}
impl<'de> Deserialize<'de> for ModelName 
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}


#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelSize
{
    W0_6B,
    W1_7B,
    W4B,
    W7B
}
impl FromStr for ModelSize
{
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> 
    {
        match s.to_lowercase().as_str()
        {
            "0.6b" => Ok(ModelSize::W0_6B),
            "1.7b" => Ok(ModelSize::W1_7B),
            "4b" => Ok(ModelSize::W4B),
            "7b" => Ok(ModelSize::W7B),
            _ => Err("Значение параметра веса модели неверно! Возможны только значения 0.6b, 1.7b, 4b, 7b".to_owned())
        }
    }
}
impl Serialize for ModelSize 
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self 
        {
            ModelSize::W0_6B => serializer.serialize_str("0.6B"),
            ModelSize::W1_7B => serializer.serialize_str("1.7B"),
            ModelSize::W4B => serializer.serialize_str("4B"),
            ModelSize::W7B => serializer.serialize_str("7B"),
        }
    }
}
impl<'de> Deserialize<'de> for ModelSize 
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}


#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Settings
{
    pub system_prompt: String,
    pub model_name: ModelName,
    pub model_size: ModelSize,
    pub model_settings: Option<ModelSettings>
}
impl Default for Settings
{
    fn default() -> Self 
    {
        Self 
        { 
            system_prompt: "Ты корректор текстов на русском языке, находи ошибки в тексте, смысловых ошибок в тексте не будет, склонения слов не проверяй, возможны только орфографические ошибки. Если есть ошибки выдавай их в виде json объекта: {{ start_index: number, end_index: number, description: string }} где start_index начальный индекс слова с ошибкой измеряемый посимвольно с начала строки, end_index конечный индекс слова с ошибкой измеряемый посимвольно с начала строки, description описание ошибки, если ошибок несколько выводи как массив объектов. Если ошибок нет, не выводи ничего.".to_owned(),
            model_name: ModelName::Qwen,
            model_size: ModelSize::W0_6B,
            model_settings: Some(ModelSettings::default())
        }
    }
}
impl Settings
{
    pub fn load() -> anyhow::Result<Self>
    {
        let settings = utilites::deserialize("settings.toml", false, utilites::Serializer::Toml)?;
        Ok(settings)
    }
    pub fn save(&self)
    {
        let r = utilites::serialize(self, "settings.toml", false, utilites::Serializer::Toml);
        if let Err(e) = r
        {
            logger::error!("{}", e);
        }
    }
}

pub trait ModelFromSettings: Sized
{
    fn from_settings(settings: Settings) -> anyhow::Result<Self>;
}
#[cfg(test)]
mod tests
{
    #[test]
    fn test()
    {
        let _ = logger::StructLogger::new_default();
        let settings = super::Settings::default();
        settings.save();
    }
}