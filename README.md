# 🔍 Previsão de Churn de Clientes em Telecomunicações

Este projeto aplica aprendizado de máquina para prever se um cliente de uma empresa de telecomunicações irá cancelar o serviço (*churn*).  
O sistema realiza pré-processamento automatizado, preenchimento inteligente de dados faltantes e **classificação com Random Forest**, tudo integrado em uma **pipeline do Scikit-Learn**.

---

## 🚀 Tecnologias Utilizadas
- Python 🐍  
- Pandas, NumPy  
- Scikit-Learn (Pipeline, ColumnTransformer, RandomForestClassifier)  
- Machine Learning para classificação binária  

---

## ⚙️ Etapas Principais
1. **Correção de dados inconsistentes** (`TotalCharges` convertido e completado com regressão linear)  
2. **Pré-processamento automatizado**:
   - Imputação de valores ausentes  
   - Padronização de variáveis numéricas  
   - Codificação *One-Hot* de variáveis categóricas  
3. **Criação de Pipeline completa** integrando todas as etapas  
4. **Treinamento e avaliação do modelo Random Forest**  

---

## 📊 Resultados
O modelo gera métricas como **Precisão**, **Recall**, **F1-score** e **Acurácia**, conforme o relatório do `classification_report()`.

Exemplo de saída:

