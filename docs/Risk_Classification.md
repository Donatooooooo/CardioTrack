# **Risk Classification**

## **1. Purpose**

This document describes the risk classification of a Software as a Medical Device (SaMD) designed to identify the presence or absence of signs of heart failure through a **binary classification based on clinical data**.
The classification is developed using the **IMDRF SaMD Risk Categorization** framework and additional European regulatory references (AI Act, MDR).

---

## **2. Intended Use**

The system performs a **binary classification** task based on clinical data from individual patients, aiming to predict the presence or absence of heart disease. Specifically, the model analyzes each patient’s clinical features and risk factors to identify the potential presence of heart failure.

The model outputs two possible classification results:

* **Positive:** when the patient shows indicators compatible with heart failure.
* **Negative:** when no signs of the condition are detected.

### **2.1 Clinical Role**

* The software output is intended as a **Clinical Decision Support (CDS)** tool.
* The intended user is a **qualified medical professional**.
* The software **does not perform diagnosis**, **does not make autonomous therapeutic decisions**, and **is not intended for use in emergency settings**.
* The information provided supports—but does not replace—clinical judgement.

---

## **3. IMDRF SaMD Risk Categorization**

The IMDRF framework evaluates two key dimensions:

1. **The significance of the information provided by the software**
2. **The severity of the clinical condition addressed**

### **3.1 Significance of the Information – *Treat/Diagnose***

**Rationale:**

* The software provides a **binary risk classification** that may influence clinical decisions such as follow-up, diagnostic investigation, or changes in patient management.
* The output goes beyond merely describing clinical status (“inform” level), contributing instead to medical decision-making.
* As the system supports decisions relevant to diagnosis and treatment, it falls within the **Treat/Diagnose** category of the IMDRF framework.

### **3.2 Severity of the Clinical Condition – *Serious***

**Rationale:**

* Heart failure is a serious medical condition with potentially significant complications.
* The system is not intended for emergency use, does not initiate immediate life-saving actions, and operates within routine or preventive clinical care.
* The presence of a medical professional mitigates the risk of immediate harm due to software errors.
* In the intended use context, the condition is therefore appropriately classified as **Serious**, not “Critical”.

### **3.3 IMDRF Classification Result**

| Significance   | Condition | IMDRF Category |
| -------------- | --------- | -------------- |
| Treat/Diagnose | Serious   | **III**        |

---

## **4. AI Act – Probable Classification as High-Risk AI**

According to the **Regulation (EU) 2024/1689 (AI Act)**, artificial intelligence systems used as medical devices or as components of medical devices regulated under the MDR/IVDR are included among **High-Risk AI Systems**, as listed in Annex III.

For the system under consideration:

* it meets the MDR definition of **SaMD**;
* it supports clinically relevant decisions;
* it may influence patient management concerning a serious medical condition.

Therefore, the system can be **reasonably considered a High-Risk AI System** under the AI Act.
This is not a definitive classification—the formal designation will depend on MDR processes and final technical documentation—but it represents a consistent regulatory interpretation based on the software’s intended purpose and domain.

---

## **5. Conclusion**

The software is classified as:

* **SaMD, IMDRF Category III**, based on:

  * information of the **Treat/Diagnose** type;
  * management of a condition categorized as **Serious**.

This classification does not represent the final MDR class but provides a robust basis for risk assessment and regulatory positioning, including the likely classification as **High-Risk AI** under the AI Act.

---

## **6. MDR (EU) – Additional Note**

IMDRF categorization does not directly determine the MDR class but offers a helpful conceptual framework.

The European MDR classification will be established through:

* **MDR 2017/745**, Annex VIII
* **Rule 11**, specific to medical device software
* **MDCG 2019-11**, interpretative guidance

Based on the system’s functionality (clinical classification supporting diagnosis/prognosis), an assignment of at least **Class IIa** is likely.
However, the final class will depend on the risk evaluation and the risk-control measures implemented.


