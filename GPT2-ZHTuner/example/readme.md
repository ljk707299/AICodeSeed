# GPT2-ZHTuner 示例程序说明

## 目录结构
- `test01.py`: 基础文本生成示例
- `test02.py`: 歌词生成专用示例
- `test03.py`: 古文生成示例
- `test04.py`: 对联生成示例
- `test05.py`: 诗歌生成示例

## 使用说明
1. 确保已安装所有依赖包：`pip install -r requirements.txt`
2. 模型文件存放于项目内 <mcfolder name="model" path="/Users/lijiakai/code/ai_study/AICodeSeed/GPT2-ZHTuner/model"></mcfolder> 目录
3. 所有示例程序均已配置为使用本地模型，无需额外下载
4. 运行示例：`python test01.py`

## 模型说明
- 所有示例使用的GPT2模型均来自UER开源项目
- 不同示例对应不同训练领域的模型变体
- 模型路径已统一配置为项目内相对路径

## 注意事项
- 如遇模型加载问题，请检查模型目录完整性
- 首次运行可能需要几分钟时间进行模型初始化
- 生成结果质量受输入提示词影响，建议提供清晰的创作方向