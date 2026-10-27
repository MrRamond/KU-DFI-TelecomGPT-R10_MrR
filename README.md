# TelecomGPT-R1 — integração local

Integração para baixar e executar [KU-DFI/TelecomGPT-R1](https://huggingface.co/KU-DFI/TelecomGPT-R1), modelo de aproximadamente 27 bilhões de parâmetros. Este repositório contém os scripts de integração. Os pesos (~55 GB) são obtidos do Hugging Face e ficam em uma pasta ignorada pelo Git.

## Requisitos

- Python 3.10 ou superior; Python 3.11 recomendado.
- Git e acesso ao Hugging Face para o download inicial.
- Pelo menos 70 GB livres em disco para os pesos e margem adicional para dependências.
- GPU compatível com bfloat16. O autor descreve execução BF16 em A100/H100 de 80 GB ou equivalente. O consumo depende do contexto, da carga e da distribuição entre GPUs.
- PyTorch com CUDA/ROCm adequado ao equipamento. Instale conforme o [seletor oficial do PyTorch](https://pytorch.org/get-started/locally/).
- Windows: os comandos abaixo usam PowerShell; para servidores Linux, substitua o caminho do Python por `.venv/bin/python`.

## Instalação no Windows

Execute em uma pasta local autorizada pela TI:

```powershell
git clone https://github.com/MrRamond/KU-DFI-TelecomGPT-R10_MrR.git
cd KU-DFI-TelecomGPT-R10_MrR
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Instale a versão de PyTorch compatível com sua GPU usando o seletor oficial acima. Depois:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_model.py --check
```

O comando `--check` verifica importações e suporte da GPU, sem baixar ou carregar o modelo. Não garante disponibilidade de memória para inferência.

## Download e primeira pergunta

O próximo comando transfere aproximadamente 55 GB:

```powershell
.\.venv\Scripts\python.exe download_model.py
.\.venv\Scripts\python.exe run_model.py "Explique as diferenças entre LTE-M e NB-IoT."
```

O download resolve a revisão solicitada para um commit exato antes de transferir os arquivos e registra esse SHA em `models/TelecomGPT-R1/integration-source.json`. Para repetir uma versão específica, use `download_model.py --revision SHA_DO_COMMIT`. Uma nova execução pode reaproveitar arquivos já baixados.

Para outra pasta:

```powershell
.\.venv\Scripts\python.exe download_model.py --output D:\Modelos\TelecomGPT-R1
.\.venv\Scripts\python.exe run_model.py --model-dir D:\Modelos\TelecomGPT-R1 "O que é uma APN?" --max-new-tokens 1024
```

Pastas fora de `models/` devem permanecer fora do controle de versão. Não adicione pesos ao Git.

## Uso offline

Após concluir o download, `run_model.py` carrega exclusivamente arquivos locais (`local_files_only=True`). As perguntas são processadas na máquina que executa o script. O script oferece geração de texto por terminal; não inicia servidor ou interface web.

## Arquitetura e limitações

O checkpoint declara `Qwen3_5ForConditionalGeneration`; essa classe é usada explicitamente, com BF16, conforme a [documentação da arquitetura](https://huggingface.co/docs/transformers/model_doc/qwen3_5). A integração limita pergunta mais resposta a 8192 tokens. O script usa distribuição automática entre dispositivos; memória insuficiente pode causar descarregamento para RAM e grande lentidão, ou erro de memória.

Dependências possuem faixas de versão; este projeto não é um ambiente de produção com versões totalmente fixadas. Depois de validar no servidor de destino, registre as versões instaladas para reproduzir esse ambiente.

Validação realizada na preparação: sintaxe Python e fluxo de argumentos sem download. A inferência completa deve ser validada no equipamento de destino, com os pesos e GPU adequados.

## Origem e licença

Modelo desenvolvido por KU/DFI. A página de origem informa licença Apache-2.0; consulte o [model card original](https://huggingface.co/KU-DFI/TelecomGPT-R1) e preserve os avisos aplicáveis em redistribuições. Estes scripts são uma integração independente e não alteram nem treinam os pesos.
