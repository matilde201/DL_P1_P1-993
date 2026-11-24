# Classificação de Tumores Cerebrais em Ressonância Magnética (MRI)

**Unidade Curricular:** Deep Learning Aplicado (Mestrado — UTAD, Ano Letivo 2026/2027)  
**Trabalho Prático 1 (P1):** Protocolos 1 a 5

---

## 1. Caracterização do Problema e Dataset (Aula 1 e 2)

O objetivo deste projeto consiste no desenvolvimento, avaliação e interpretação de modelos de Deep Learning para a classificação de ressonâncias magnéticas cerebrais em 4 classes: *glioma*, *meningioma*, *notumor* e *pituitary*.

### Tabela 2.4 (Aula 1) — Metadados do Dataset

| Campo | Resposta |
| :--- | :--- |
| **Dataset e URL** | **Brain Tumor MRI Dataset** (Masoud Nickparvar)<br>https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset |
| **N.º de imagens / n.º de classes** | 7 200 imagens / 4 classes |
| **Distribuição por classe (%)** | • *glioma*: 25.0% (1 800 imagens)<br>• *meningioma*: 25.0% (1 800 imagens)<br>• *notumor*: 25.0% (1 800 imagens)<br>• *pituitary*: 25.0% (1 800 imagens) |
| **Licença** | CC0: Public Domain |
| **Existe identificador de doente, local ou sessão? Qual?** | Não existe identificador explícito de doente nos metadados. Foram removidos duplicados exatos por MD5 antes da divisão. Sem identificadores de doente, não conseguimos confirmar se cortes diferentes do mesmo doente ficaram no mesmo conjunto. |
| **Porque é um problema interessante?** | A deteção e diferenciação precoce de tumores cerebrais auxilia o planeamento clínico e cirúrgico, reduzindo a variabilidade interobservador em imagiologia médica. |

---

### Tabela 2.1 (Aula 2) — Caracterização do Dataset

| Classe | N.º imagens | % | Dimensão típica | Observações (qualidade, duplicados, marcas) |
| :--- | :---: | :---: | :---: | :--- |
| **glioma** | 1 800 | 25.0% | (512, 512) | Bom contraste; contém 14 ficheiros duplicados (MD5). |
| **meningioma** | 1 800 | 25.0% | (512, 512) | Boa nitidez; contém 16 ficheiros duplicados (MD5). |
| **notumor** | 1 800 | 25.0% | (225, 225) | Dimensão reduzida; contém 119 ficheiros duplicados (MD5). |
| **pituitary** | 1 800 | 25.0% | (512, 512) | Foco hipofisário; contém 38 ficheiros duplicados (MD5). |

*Nota complementar:* Total de imagens com o mesmo MD5 em todo o dataset: **187 duplicados** (resultando em **7 013 imagens únicas** após limpeza).

---

## 2. Divisão e Protocolo Experimental (Aula 2)

* **Divisão dos Dados:** 70% Treino (4 909 imagens), 15% Validação (1 052 imagens) e 15% Teste (1 052 imagens), estratificada por classe com semente fixa (`random_state=42`) e guardada em `data/splits.csv`.
* **Verificação de duplicados:** o ponto 2.2 compara os MD5 entre treino, teste e validação. As três interseções têm de ser zero.
* **Isolamento do Teste:** O conjunto de teste foi mantido intocado durante o ajuste de hiperparâmetros e decisão de paragem do treino.

---

## 3. Desempenho e Comparação de Modelos

### Tabela 2.4 (Aula 2) — Desempenho na Melhor Época (CNN Baseline)

| Época com menor val_loss | Acc treino | Acc validação | Classificador maioritário | Diagnóstico (underfitting / overfitting / adequado) |
| :---: | :---: | :---: | :---: | :--- |
| 28 | 93.22% | 92.11% | 25.47% | **Adequado** (bom ajuste, sem overfitting severo; menor val_loss: 0.2732) |

---

### Avaliação final no conjunto de teste (Aula 4)

O modelo de transferência e o limiar para *meningioma* são escolhidos na validação. O teste usa a divisão de `data/splits.csv` e gera a matriz de confusão, o relatório por classe e a ROC/AUC. A avaliação anterior de 1 080 imagens correspondia à divisão preliminar. Os resultados do teste final e os mapas Grad-CAM estão pendentes de execução.

---

### Tabela 2.4 (Aula 3) — Comparação de Modelos (Transfer Learning vs Baseline)

| Modelo | Parâmetros treináveis | Melhor época | Acc/métrica val | Tempo total |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (aula 2)** | 389,444 | 28 | 92.11% (val_loss: 0.2732) | A medir na nova execução |
| **Extração de características** | 5,124 | 18 | 88.12% (val_loss: 0.3410) | 404.0 s |
| **Fine-tuning** | — | — | Pendente de execução | — |

---

### Tabela 2.5 (Aula 3) — Verificação da Base Pré-treinada com IA

| Proposta da IA (código ou afirmação) | Funcionou / estava correta? (S/N) | O que estava errado ou em falta | Como verificou (output, shape, métrica) |
| :--- | :---: | :--- | :--- |
| **Parâmetros e Resolução:** `EfficientNetB0` (`include_top=False`) tem **4,049,571** parâmetros, contra **23,587,712** da `ResNet50` e **27,820,128** da `ConvNeXtTiny`, todas com entrada `(None, 224, 224, 3)`. | **S** | Nada errado; contagens verificadas com precisão exata. | Instanciação das três redes no Colab imprimindo `m.input_shape` e `m.count_params()`. |
| **Pré-processamento:** `EfficientNetB0` inclui normalização interna (espera píxeis `[0, 255]`), enquanto `ResNet50` exige `resnet50.preprocess_input` (RGB para BGR e subtração da média). | **S** | Se trocássemos diretamente para `ResNet50` sem alterar a pipeline de dados, a normalização ficaria em falta. | Documentação oficial do Keras e inspeção das camadas da base. |
| **Adequação:** A `EfficientNetB0` tem menos parâmetros do que as outras duas bases comparadas. | **S** | A comparação de desempenho exigiria treinar as três bases na mesma divisão. | Contagem de parâmetros com `m.count_params()`. |
