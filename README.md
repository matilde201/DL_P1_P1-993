# Classificação de tumores cerebrais em ressonância magnética

Deep Learning Aplicado · UTAD · P1-993 · Protocolos 1 a 5

## Dados e divisão

O projeto usa o [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), com 7 200 imagens e quatro classes de 1 800 imagens cada (*glioma*, *meningioma*, *notumor* e *pituitary*). A licença indicada para o dataset é CC0.

Removemos 187 duplicados exatos por MD5. A divisão estratificada com seed 42 ficou com 4 909 imagens de treino, 1 052 de validação e 1 052 de teste. As três interseções de MD5 foram zero. Sem identificadores de doente, podem existir cortes diferentes do mesmo doente em conjuntos distintos.

## Execução

1. Abrir `DL_P1_P1_993.ipynb` no Colab e selecionar um ambiente com GPU.
2. Para reutilizar a divisão, colocar o manifesto desta execução em `data/splits.csv` no ambiente do Colab.
3. Executar as células por ordem. O notebook obtém as imagens via KaggleHub, verifica a divisão, treina os modelos e guarda os checkpoints em `modelos/`. Os logs do TensorBoard ficam em `logs/`.

## Validação

Os valores correspondem à época com menor perda de validação para cada modelo.

| Modelo | Parâmetros treináveis | Época | Exatidão | Perda | Tempo |
|---|---:|---:|---:|---:|---:|
| CNN de raiz | 389 444 | 28 | 90,87% | 0,3007 | 279,7 s |
| Extração de características | 5 124 | 10 | 84,41% | 0,3901 | 264,4 s |
| Fine-tuning | 1 356 084 | 20 | 88,21% | 0,2919 | 371,3 s |

A experiência com dropout na baseline obteve 91,35% de exatidão e perda de 0,2666 na época 29. Entre as variantes de transferência, o fine-tuning teve menor perda de validação e foi o modelo avaliado no teste.

## Teste final

Em 1 052 imagens, o fine-tuning obteve **89,64% de exatidão**, **F1 macro de 0,8953** e **AUC macro OvR de 0,9868**.

| Classe | Precisão | Recall | F1 | Imagens |
|---|---:|---:|---:|---:|
| glioma | 0,8841 | 0,9104 | 0,8971 | 268 |
| meningioma | 0,8874 | 0,7351 | 0,8041 | 268 |
| notumor | 0,9719 | 0,9603 | 0,9661 | 252 |
| pituitary | 0,8525 | 0,9848 | 0,9139 | 264 |

Escolhemos o limiar 0,25 para meningioma na validação, dando prioridade a reduzir falsos negativos. No teste, o recall para essa classe com o limiar foi 82,46% e a precisão foi 81,55%. O notebook contém a matriz de confusão, as curvas ROC e a análise dos quatro Grad-CAM.

O uso de IA está registado em `AI_LOG.md`.
