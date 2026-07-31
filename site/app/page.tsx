"use client";

import { useMemo, useState } from "react";
import Image from "next/image";

type Language = "en" | "zh";
type Level = "Class" | "Order" | "Family" | "Genus" | "Species";

type ModelStage = {
  id: string;
  family: "Flat" | "Hierarchical";
  samples: number;
  split: [number, number, number];
  output: string;
  init: string;
  note: { en: string; zh: string };
};

const stages: ModelStage[] = [
  {
    id: "F-C",
    family: "Flat",
    samples: 3522,
    split: [2464, 529, 529],
    output: "Class",
    init: "ImageNet",
    note: {
      en: "Class-only baseline. Its test set is larger than the final species subset, so its accuracy is not a controlled endpoint comparison with H-COFGS.",
      zh: "仅预测 Class 的基线。它的测试集大于最终 Species 子集，因此不能把其准确率与 H-COFGS 当作受控端点比较。",
    },
  },
  {
    id: "F-G",
    family: "Flat",
    samples: 3472,
    split: [2430, 521, 521],
    output: "Genus",
    init: "ImageNet",
    note: {
      en: "Genus is predicted directly; upper ranks are recovered by deterministic taxonomy lookup. It shares the source data and fixed membership with H-COFG.",
      zh: "直接预测 Genus，上层级由确定性的 taxonomy lookup 得到。它与 H-COFG 共享来源数据和固定划分成员。",
    },
  },
  {
    id: "F-S",
    family: "Flat",
    samples: 1456,
    split: [1018, 219, 219],
    output: "Species",
    init: "ImageNet",
    note: {
      en: "Species is predicted directly; upper ranks are looked up from the predicted species. This is the matched flat comparator for H-COFGS.",
      zh: "直接预测 Species，上层级由预测 Species 反查得到。它是 H-COFGS 的配对 flat comparator。",
    },
  },
  {
    id: "H-CO",
    family: "Hierarchical",
    samples: 3506,
    split: [2454, 526, 526],
    output: "Class -> Order",
    init: "F-C backbone",
    note: {
      en: "The EfficientNet-B0 backbone starts from F-C weights and remains trainable. Stage-specific classifier heads are newly initialized.",
      zh: "EfficientNet-B0 backbone 从 F-C 权重开始并继续训练；本阶段 classifier heads 重新初始化。",
    },
  },
  {
    id: "H-COF",
    family: "Hierarchical",
    samples: 3495,
    split: [2445, 525, 525],
    output: "Class -> Order -> Family",
    init: "H-CO backbone",
    note: {
      en: "Predicted parent probabilities enter downstream heads. Ground-truth parent masks constrain the training loss, not the head inputs.",
      zh: "预测得到的 parent probabilities 进入下游 heads；ground-truth parent masks 约束训练 loss，而不是替换 head inputs。",
    },
  },
  {
    id: "H-COFG",
    family: "Hierarchical",
    samples: 3472,
    split: [2430, 521, 521],
    output: "Class -> Order -> Family -> Genus",
    init: "H-COF backbone",
    note: {
      en: "This stage can be compared directly with F-G because both use the same 521-image test membership.",
      zh: "本阶段可以与 F-G 直接比较，因为二者使用相同的 521 张测试图像成员。",
    },
  },
  {
    id: "H-COFGS",
    family: "Hierarchical",
    samples: 1456,
    split: [1018, 219, 219],
    output: "Class -> Order -> Family -> Genus -> Species",
    init: "H-COFG backbone",
    note: {
      en: "Greedy decoding uses each predicted parent to mask taxonomically invalid children at the next rank.",
      zh: "Greedy decoding 使用每一级预测 parent 屏蔽下一层级中不符合 taxonomy 的 children。",
    },
  },
];

const comparison: Array<{ level: Level; hierarchical: number; flat: number }> = [
  { level: "Class", hierarchical: 99.5, flat: 98.6 },
  { level: "Order", hierarchical: 98.6, flat: 91.8 },
  { level: "Family", hierarchical: 97.7, flat: 90.4 },
  { level: "Genus", hierarchical: 97.7, flat: 90.0 },
  { level: "Species", hierarchical: 69.4, flat: 69.4 },
];

const references = [
  {
    key: "Round et al., 1990",
    title: "The Diatoms: Biology and Morphology of the Genera",
    role: "Taxonomic and morphological background",
    roleZh: "硅藻 taxonomy 与 morphology 背景",
    href: "https://assets.cambridge.org/97805217/14693/frontmatter/9780521714693_frontmatter.pdf",
  },
  {
    key: "Silla & Freitas, 2011",
    title: "A survey of hierarchical classification across different application domains",
    role: "Hierarchical-classification framework",
    roleZh: "层级分类框架",
    href: "https://doi.org/10.1007/s10618-010-0175-9",
  },
  {
    key: "Caruana, 1997",
    title: "Multitask Learning",
    role: "Shared representations and multi-task supervision",
    roleZh: "共享表征与 multi-task supervision",
    href: "https://doi.org/10.1023/A:1007379606734",
  },
  {
    key: "Tan & Le, 2019",
    title: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks",
    role: "EfficientNet-B0 backbone",
    roleZh: "EfficientNet-B0 backbone 来源",
    href: "https://proceedings.mlr.press/v97/tan19a.html",
  },
  {
    key: "Lin et al., 2017",
    title: "Focal Loss for Dense Object Detection",
    role: "Focal-loss formulation",
    roleZh: "Focal loss 公式来源",
    href: "https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html",
  },
  {
    key: "Bertinetto et al., 2020",
    title: "Making Better Mistakes: Leveraging Class Hierarchies With Deep Networks",
    role: "Hierarchy-aware error severity",
    roleZh: "考虑层级结构的错误严重程度",
    href: "https://openaccess.thecvf.com/content_CVPR_2020/html/Bertinetto_Making_Better_Mistakes_Leveraging_Class_Hierarchies_With_Deep_Networks_CVPR_2020_paper.html",
  },
  {
    key: "Boone-Sifuentes et al., 2022",
    title: "A Mask-based Output Layer for Multi-level Hierarchical Classification",
    role: "Taxonomy-constrained output masking",
    roleZh: "Taxonomy-constrained output masking",
    href: "https://doi.org/10.1145/3511808.3557534",
  },
];

const copy = {
  en: {
    nav: ["Pipeline", "Models", "Results", "Reproducibility", "References"],
    kicker: "Hierarchical diatom classification",
    subtitle:
      "A multi-level taxonomic image classifier, presented with the experiment boundaries that make each result interpretable.",
    paper: "Read paper",
    code: "View code",
    overview: "What was tested",
    overviewText:
      "DiatomCascadeNet compares flat classifiers with cascaded hierarchical models over Class, Order, Family, Genus, and Species. This companion site presents the study design, matched comparisons, and supporting literature.",
    pipeline: "Experiment pipeline",
    pipelineText:
      "The split is a saved artifact, not a fresh random operation during evaluation. Training and model selection use train and validation; the final report uses test once.",
    models: "Seven-model experiment map",
    modelsText:
      "Select a stage to inspect its dataset, initialization, outputs, and role in the progressive experiment design.",
    split: "Fixed split",
    output: "Direct outputs",
    init: "Initialization",
    design: "Stage design",
    results: "Matched endpoint comparison",
    resultsText:
      "H-COFGS and F-S use the same 219-image test manifest. Both reach 69.4% species accuracy; the hierarchical model retains stronger upper-rank predictions.",
    locality: "Equal species accuracy, different error locality",
    localityText:
      "These values were independently recomputed from the stored per-sample predictions and filtered taxonomy labels.",
    reproducibility: "Reproducibility boundaries",
    references: "Supporting literature",
    referencesText:
      "These independently verified sources support the public description of the model architecture, hierarchical classification, loss design, and taxonomy-aware interpretation.",
  },
  zh: {
    nav: ["流程", "模型", "结果", "可复现性", "文献"],
    kicker: "层级硅藻图像分类",
    subtitle: "一个多层级硅藻图像分类研究，并同时呈现解释每项结果所必需的实验边界。",
    paper: "阅读论文",
    code: "查看代码",
    overview: "研究测试了什么",
    overviewText:
      "DiatomCascadeNet 比较 flat classifiers 与 Class、Order、Family、Genus、Species 五级 cascaded hierarchical models。本网站面向外部读者呈现实验设计、配对比较和支持性文献。",
    pipeline: "实验流程",
    pipelineText:
      "Split 是被保存的实验工件，不是在 evaluation 时重新随机生成。训练和模型选择只使用 train 与 validation；最终报告只使用一次 test。",
    models: "七模型实验地图",
    modelsText:
      "选择一个阶段，查看它的数据集、初始化、输出以及在渐进实验设计中的作用。",
    split: "固定划分",
    output: "直接输出",
    init: "初始化",
    design: "阶段设计",
    results: "配对端点比较",
    resultsText:
      "H-COFGS 与 F-S 使用同一个 219-image test manifest。二者 Species accuracy 均为 69.4%，但 hierarchical model 在上层 taxonomy ranks 保持更高准确率。",
    locality: "Species accuracy 相同，错误的分类距离不同",
    localityText:
      "这些数值已根据逐样本预测记录和 filtered taxonomy labels 独立重新计算。",
    reproducibility: "可复现性边界",
    references: "支持性文献",
    referencesText:
      "以下独立核实的来源支持本网站对模型架构、层级分类、loss 设计和 taxonomy-aware interpretation 的公开说明。",
  },
};

function EvidenceTag({ tone, children }: { tone: "artifact" | "recomputed"; children: React.ReactNode }) {
  return <span className={`evidence-tag ${tone}`}>{children}</span>;
}

export default function Home() {
  const [language, setLanguage] = useState<Language>("en");
  const [selectedId, setSelectedId] = useState("H-COFGS");
  const t = copy[language];
  const selected = useMemo(() => stages.find((stage) => stage.id === selectedId) ?? stages[6], [selectedId]);

  const rankName = (level: Level) => language === "en" ? level : ({ Class: "纲 Class", Order: "目 Order", Family: "科 Family", Genus: "属 Genus", Species: "种 Species" }[level]);
  const pipelineSteps = language === "en"
    ? [
        ["A", "Raw source", "4,881 microscopy images + curated labels"],
        ["B", "Taxonomy cleaning", "3,522 complete, resolved records"],
        ["C", "Model filtering", "Minimum 10 images at the deepest rank"],
        ["D", "Fixed manifests", "70% train / 15% validation / 15% test"],
        ["E", "Preflight", "3,522 referenced images readable"],
        ["F", "Training", "Validation selects the checkpoint"],
        ["G", "Final evaluation", "Test manifest + stored JSON artifact"],
      ]
    : [
        ["A", "原始数据", "4,881 张显微图像与人工整理标签"],
        ["B", "Taxonomy 清理", "3,522 条完整且冲突已处理的记录"],
        ["C", "模型专属过滤", "最深层级每个 taxon 至少 10 张图像"],
        ["D", "固定 manifests", "70% train / 15% validation / 15% test"],
        ["E", "图像预检", "3,522 个引用图像均可读取"],
        ["F", "训练", "使用 validation 选择 checkpoint"],
        ["G", "最终评估", "固定 test manifest 与 JSON 结果工件"],
      ];
  const reproducibilityItems = language === "en"
    ? [
        ["artifact", "fixed", "Split membership", "Training, validation, and test membership is saved once for each model variant and reused during evaluation."],
        ["artifact", "progressive", "Backbone initialization", "Each hierarchical stage inherits and continues updating the preceding EfficientNet-B0 backbone. Stage-specific classifier heads are initialized for the current outputs."],
        ["artifact", "constrained", "Hierarchy masks", "Training masks constrain valid loss candidates. Greedy inference constrains children using the predicted parent at each rank."],
        ["recomputed", "deterministic", "Taxonomy derivation", "The hierarchy is deterministically extracted from curated labels; it is not learned from image pixels or optimized on test performance."],
      ]
    : [
        ["artifact", "固定", "Split 成员", "每个模型变体只生成一次 train、validation 和 test 成员，并在 evaluation 阶段重复使用。"],
        ["artifact", "渐进", "Backbone 初始化", "每个 hierarchical stage 继承并继续更新前一阶段的 EfficientNet-B0 backbone；当前阶段所需的 classifier heads 单独初始化。"],
        ["artifact", "约束", "Hierarchy masks", "训练 masks 约束有效 loss 候选；greedy inference 根据每一级预测 parent 约束下一层 children。"],
        ["recomputed", "确定性", "Taxonomy 的生成", "层级结构由人工整理 labels 确定性提取，不从图像像素中学习，也不针对 test performance 优化。"],
      ];

  return (
    <main lang={language === "zh" ? "zh-CN" : "en"}>
      <header className="site-header">
        <a className="wordmark" href="#top" aria-label="DiatomCascadeNet home">
          <span className="wordmark-mark">DC</span>
          <span>DiatomCascadeNet</span>
        </a>
        <nav aria-label="Primary navigation">
          {[
            ["#pipeline", t.nav[0]],
            ["#models", t.nav[1]],
            ["#results", t.nav[2]],
            ["#reproducibility", t.nav[3]],
            ["#references", t.nav[4]],
          ].map(([href, label]) => (
            <a href={href} key={href}>{label}</a>
          ))}
        </nav>
        <div className="language-switch" role="group" aria-label="Language">
          <button className={language === "en" ? "active" : ""} onClick={() => setLanguage("en")} aria-pressed={language === "en"}>EN</button>
          <button className={language === "zh" ? "active" : ""} onClick={() => setLanguage("zh")} aria-pressed={language === "zh"}>中文</button>
        </div>
      </header>

      <section className="hero" id="top">
        <Image
          src="https://upload.wikimedia.org/wikipedia/commons/b/b3/Naviculoid_diatom.jpg"
          alt="Light micrograph of two living naviculoid diatoms"
          fill
          priority
          sizes="100vw"
          unoptimized
        />
        <div className="hero-shade" />
        <div className="hero-content">
          <p className="eyebrow">{t.kicker}</p>
          <h1>DiatomCascadeNet</h1>
          <p className="hero-subtitle">{t.subtitle}</p>
          <div className="hero-actions">
            <a className="primary-action" href="https://arxiv.org/abs/2512.06613" target="_blank" rel="noreferrer">{t.paper} <span aria-hidden="true">↗</span></a>
            <a className="secondary-action" href="https://github.com/DinaberryPi/DiatomCascadeNet-public" target="_blank" rel="noreferrer">{t.code} <span aria-hidden="true">↗</span></a>
          </div>
        </div>
        <p className="image-credit">
          Micrograph: <a href="https://commons.wikimedia.org/wiki/File:Naviculoid_diatom.jpg" target="_blank" rel="noreferrer">Djpmapfer, CC BY-SA 4.0</a>
        </p>
      </section>

      <section className="overview-band" aria-labelledby="overview-title">
        <div className="section-inner overview-grid">
          <div>
            <p className="section-number">01</p>
            <h2 id="overview-title">{t.overview}</h2>
            <p className="lead">{t.overviewText}</p>
          </div>
          <dl className="headline-stats">
            <div><dt>7</dt><dd>{language === "en" ? "model variants" : "模型变体"}</dd></div>
            <div><dt>5</dt><dd>{language === "en" ? "taxonomic ranks" : "taxonomy 层级"}</dd></div>
            <div><dt>1,456</dt><dd>{language === "en" ? "final-subset images" : "最终子集图像"}</dd></div>
            <div><dt>82</dt><dd>{language === "en" ? "species" : "Species 类别"}</dd></div>
          </dl>
        </div>
      </section>

      <section className="section" id="pipeline" aria-labelledby="pipeline-title">
        <div className="section-inner">
          <p className="section-number">02</p>
          <div className="section-heading">
            <h2 id="pipeline-title">{t.pipeline}</h2>
            <p>{t.pipelineText}</p>
          </div>
          <div className="pipeline" aria-label="Experiment pipeline">
            {pipelineSteps.map(([index, title, detail]) => (
              <div className="pipeline-step" key={index}>
                <span>{index}</span>
                <div><strong>{title}</strong><small>{detail}</small></div>
              </div>
            ))}
          </div>
          <div className="evidence-line">
            <EvidenceTag tone="artifact">manifest-verified</EvidenceTag>
            <span>{language === "en" ? "0 overlaps across train / validation / test in all seven model sets" : "七组模型的 train / validation / test 之间均为 0 个重叠样本"}</span>
          </div>
        </div>
      </section>

      <section className="section models-section" id="models" aria-labelledby="models-title">
        <div className="section-inner">
          <p className="section-number">03</p>
          <div className="section-heading">
            <h2 id="models-title">{t.models}</h2>
            <p>{t.modelsText}</p>
          </div>
          <div className="stage-tabs" role="tablist" aria-label="Model stages">
            {stages.map((stage) => (
              <button
                key={stage.id}
                role="tab"
                aria-selected={selected.id === stage.id}
                className={selected.id === stage.id ? "active" : ""}
                onClick={() => setSelectedId(stage.id)}
              >
                <span>{stage.id}</span>
                <small>{language === "en" ? stage.family : stage.family === "Flat" ? "Flat 基线" : "Hierarchical"}</small>
              </button>
            ))}
          </div>
          <div className="stage-detail" role="tabpanel">
            <div className="stage-summary">
              <div><span>{t.split}</span><strong>{selected.split.join(" / ")}</strong><small>train / validation / test</small></div>
              <div><span>{t.output}</span><strong>{selected.output}</strong><small>{selected.samples.toLocaleString()} {language === "en" ? "images" : "张图像"}</small></div>
              <div><span>{t.init}</span><strong>{selected.init}</strong><small>{language === "en" ? `${selected.family} model` : `${selected.family} 模型`}</small></div>
            </div>
            <div className="stage-metrics">
              <h3>{t.design}</h3>
              <p>{selected.note[language]}</p>
            </div>
          </div>
        </div>
      </section>

      <section className="section results-section" id="results" aria-labelledby="results-title">
        <div className="section-inner">
          <p className="section-number">04</p>
          <div className="section-heading">
            <h2 id="results-title">{t.results}</h2>
            <p>{t.resultsText}</p>
          </div>
          <div className="comparison-head" aria-hidden="true"><span>{language === "en" ? "Taxonomic rank" : "Taxonomy 层级"}</span><span>{language === "en" ? "Accuracy" : "准确率"}</span></div>
          <div className="comparison-chart">
            {comparison.map((row) => (
              <div className="comparison-row" key={row.level}>
                <strong>{rankName(row.level)}</strong>
                <div className="bar-pair">
                  <div className="bar-line"><span>H-COFGS</span><div><i style={{ width: `${row.hierarchical}%` }} /></div><b>{row.hierarchical.toFixed(1)}%</b></div>
                  <div className="bar-line flat"><span>F-S</span><div><i style={{ width: `${row.flat}%` }} /></div><b>{row.flat.toFixed(1)}%</b></div>
                </div>
              </div>
            ))}
          </div>
          <div className="evidence-line">
            <EvidenceTag tone="recomputed">independently recomputed</EvidenceTag>
            <span>{language === "en" ? "Both prediction sets contain the same 219 test samples." : "两组预测记录包含相同的 219 个 test samples。"}</span>
          </div>
        </div>
      </section>

      <section className="section locality-section" aria-labelledby="locality-title">
        <div className="section-inner locality-grid">
          <div>
            <p className="section-number">05</p>
            <h2 id="locality-title">{t.locality}</h2>
            <p className="lead">{t.localityText}</p>
            <div className="evidence-line left">
              <EvidenceTag tone="recomputed">recomputed</EvidenceTag>
              <span>{language === "en" ? "67 species errors in each model" : "两个模型各有 67 个 Species 错误"}</span>
            </div>
          </div>
          <div className="distance-figure" aria-label="Taxonomic error locality comparison">
            <div className="distance-stat"><span>{language === "en" ? "Errors within the true genus" : "仍位于真实 Genus 内的错误"}</span><strong>92.5%</strong><small>H-COFGS</small><div><i style={{ width: "92.5%" }} /></div></div>
            <div className="distance-stat flat"><span>{language === "en" ? "Errors within the true genus" : "仍位于真实 Genus 内的错误"}</span><strong>67.2%</strong><small>F-S</small><div><i style={{ width: "67.2%" }} /></div></div>
            <div className="mean-distance"><span>{language === "en" ? "Mean taxonomic distance" : "平均 taxonomic distance"}</span><div><b>1.209</b><small>H-COFGS</small></div><div><b>1.955</b><small>F-S</small></div><strong>{language === "en" ? "38.2% lower" : "降低 38.2%"}</strong></div>
          </div>
        </div>
      </section>

      <section className="section audit-section" id="reproducibility" aria-labelledby="reproducibility-title">
        <div className="section-inner">
          <p className="section-number">06</p>
          <h2 id="reproducibility-title">{t.reproducibility}</h2>
          <div className="audit-grid">
            {reproducibilityItems.map(([tone, tag, title, body]) => (
              <article key={title}>
                <EvidenceTag tone={tone as "artifact" | "recomputed"}>{tag}</EvidenceTag>
                <h3>{title}</h3>
                <p>{body}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="section references-section" id="references" aria-labelledby="references-title">
        <div className="section-inner">
          <p className="section-number">07</p>
          <div className="section-heading">
            <h2 id="references-title">{t.references}</h2>
            <p>{t.referencesText}</p>
          </div>
          <ol className="reference-list">
            {references.map((reference) => (
              <li key={reference.key}>
                <span>{reference.key}</span>
                <div><a href={reference.href} target="_blank" rel="noreferrer">{reference.title} <span aria-hidden="true">↗</span></a><small>{language === "en" ? reference.role : reference.roleZh}</small></div>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <footer>
        <strong>DiatomCascadeNet</strong>
        <p>{language === "en" ? "Public companion to the DiatomCascadeNet research project." : "DiatomCascadeNet 研究项目的公开配套网站。"}</p>
        <div><a href="https://arxiv.org/abs/2512.06613">Paper</a><a href="https://github.com/DinaberryPi/DiatomCascadeNet-public">Code</a></div>
      </footer>
    </main>
  );
}
