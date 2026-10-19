# Classificação de Tumores Cerebrais em Ressonância Magnética (MRI)

**Unidade Curricular:** Deep Learning Aplicado (Mestrado — UTAD, Ano Letivo 2026/2027)  
**Trabalho Prático 1 (P1):** Protocolos 1, 2 e 3

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
| **Existe identificador de doente, local ou sessão? Qual?** | Não existe identificador explícito de doente nos metadados. Tratando-se de cortes de ressonância magnética, fatias adjacentes pertencem ao mesmo paciente. Para evitar fuga de dados (*data leakage*), foram auditados e eliminados duplicados exatos por hash MD5 antes das divisões de dados. |
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
* **Verificação de Fugas (Interseções de MD5 entre conjuntos):**
  * `len(S['train'] & S['test']) = 0`
  * `len(S['val'] & S['test']) = 0`
  * `len(S['train'] & S['val']) = 0`
* **Isolamento do Teste:** O conjunto de teste foi mantido intocado durante o ajuste de hiperparâmetros e decisão de paragem do treino.

---

## 3. Desempenho e Comparação de Modelos

### Tabela 2.4 (Aula 2) — Desempenho na Melhor Época (CNN Baseline)

| Época com menor val_loss | Acc treino | Acc validação | Classificador maioritário | Diagnóstico (underfitting / overfitting / adequado) |
| :---: | :---: | :---: | :---: | :--- |
| 29 | 93.22% | 90.97% | 25.47% | **Adequado** (bom ajuste, sem overfitting severo; menor val_loss: 0.2710) |

---

### Tabela 2.5 (Aula 2) — Avaliação Final no Conjunto de Teste

Exactidão de Teste: **90.28%** | Loss de Teste: **0.2909** (1 080 amostras avaliadas).

| Classe | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **glioma** | 0.86 | 0.93 | 0.89 | 270 |
| **meningioma** | 0.88 | 0.83 | 0.85 | 270 |
| **notumor** | 0.90 | 0.93 | 0.91 | 270 |
| **pituitary** | 0.98 | 0.93 | 0.95 | 270 |
| **accuracy** | | | **0.90** | **1 080** |
| **macro avg** | 0.90 | 0.90 | 0.90 | 1 080 |
| **weighted avg** | 0.90 | 0.90 | 0.90 | 1 080 |

---

### Tabela 2.4 (Aula 3) — Comparação de Modelos (Transfer Learning vs Baseline)

| Modelo | Parâmetros treináveis | Melhor época | Acc/métrica val | Tempo total |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (aula 2)** | 222,276 | 29 | 90.97% (val_loss: 0.2710) | ~150 s (~2.5 min) |
| **Extração de características** | 5,124 | 15 | 84.32% (val_loss: 0.3964) | 359.1 s (5.99 min) |
| **Fine-tuning** | 1,356,084 | 20 | 88.59% (val_loss: 0.3007) | 373.8 s (6.23 min) |

---

### Tabela 2.5 (Aula 3) — Verificação da Base Pré-treinada com IA

| Proposta da IA (código ou afirmação) | Funcionou / estava correta? (S/N) | O que estava errado ou em falta | Como verificou (output, shape, métrica) |
| :--- | :---: | :--- | :--- |
| **Parâmetros e Resolução:** `EfficientNetB0` (`include_top=False`) tem **4,049,571** parâmetros, contra **23,587,712** da `ResNet50` e **27,820,128** da `ConvNeXtTiny`, todas com entrada `(None, 224, 224, 3)`. | **S** | Nada errado; contagens verificadas com precisão exata. | Instanciação das três redes no Colab imprimindo `m.input_shape` e `m.count_params()`. |
| **Pré-processamento:** `EfficientNetB0` inclui normalização interna (espera píxeis `[0, 255]`), enquanto `ResNet50` exige `resnet50.preprocess_input` (RGB para BGR e subtração da média). | **S** | Se trocássemos diretamente para `ResNet50` sem alterar a pipeline de dados, a normalização ficaria em falta. | Documentação oficial do Keras e inspeção das camadas da base. |
| **Adequação:** A `EfficientNetB0` é a mais adequada por oferecer treino eficiente (~17s/época no fine-tuning) e menor suscetibilidade a overfitting em dados médicos face a modelos com mais de 23M de parâmetros. | **S** | Nada em falta; convergência estável comprovada no treino. | Curvas e métricas de validação ao longo do fine-tuning (atingindo 88.59% de exactidão na época 20). |
