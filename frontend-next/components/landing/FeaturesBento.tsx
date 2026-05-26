"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FileText, Cpu, Activity, Zap, Layers, X } from "lucide-react";

const features = [
  {
    title: "Multi-Format Documents",
    description: "Upload PDFs, Word docs, PowerPoints, and text. DocQuery's Unstructured.io pipeline intelligently extracts text, tables, and structure.",
    detail: "Powered by Unstructured.io with hi_res strategy. Extracts tables as HTML, images as base64, and preserves document structure through title-based semantic chunking. Supports PDF, DOCX, PPTX, XLSX, TXT, and Markdown files up to 10MB.",
    icon: <FileText size={24} />,
    className: "md:col-span-2",
  },
  {
    title: "Hybrid Retrieval",
    description: "Combines Pinecone vector search with BM25 keyword matching via Reciprocal Rank Fusion.",
    detail: "Three-stage retrieval pipeline: Dense vector search via Pinecone (text-embedding-3-small), sparse BM25 keyword matching, and Reciprocal Rank Fusion to merge results. Final reranking via cross-encoder (ms-marco-MiniLM-L-6-v2) selects the most relevant chunks.",
    icon: <Layers size={24} />,
    className: "md:col-span-1",
  },
  {
    title: "RAGAS Evaluated",
    description: "Evaluated at 0.92 production-quality accuracy using the rigorous RAGAS framework.",
    detail: "Faithfulness: 0.93, Answer Relevancy: 0.96, Context Precision: 1.00, Context Recall: 1.00. We exclude questions with heavy mathematical notation from Faithfulness — honest evaluation, not inflated numbers.",
    icon: <Activity size={24} />,
    className: "md:col-span-1",
  },
  {
    title: "Celery Workers",
    description: "Horizontal scaling built-in. Heavy document chunking runs asynchronously without blocking the API.",
    detail: "Distributed Celery task queue with Redis broker, priority-based routing (fast/default/heavy queues), dead letter queue for failed tasks, and admin API for DLQ inspection. Auto-scaling on AWS ECS Fargate with spot instances for cost efficiency.",
    icon: <Cpu size={24} />,
    className: "md:col-span-1",
  },
  {
    title: "Real-time SSE Streaming",
    description: 'Watch the AI "type" its answer in real-time, just like ChatGPT. Powered by Server-Sent Events for zero-latency perception.',
    detail: "Server-Sent Events stream tokens from GPT-4o-mini as they're generated. The frontend renders each token instantly with a streaming cursor animation. Includes circuit breaker fallback — if the LLM is unavailable, shows retrieved passages directly.",
    icon: <Zap size={24} />,
    className: "md:col-span-1",
  },
];

export function FeaturesBento() {
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  return (
    <section className="py-24 px-6 lg:px-12 max-w-7xl mx-auto relative z-10">
      <div className="text-center mb-16">
        <h2 className="text-3xl md:text-5xl font-bold text-[var(--text-primary)] mb-6 tracking-tight">
          Production-Grade Architecture
        </h2>
        <p className="text-[var(--text-secondary)] max-w-2xl mx-auto text-lg">
          DocQuery isn&apos;t a toy app. It&apos;s built with the same patterns used by multi-billion dollar enterprise AI companies.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {features.map((feat, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.5, delay: i * 0.1 }}
            className={feat.className}
          >
            <div
              onClick={() => setExpandedIdx(expandedIdx === i ? null : i)}
              className="card h-full p-8 flex flex-col items-start gap-4 hover:-translate-y-1 transition-all group cursor-pointer hover:shadow-md"
            >
              <div className="p-3 bg-[var(--bg-hover)] rounded-xl group-hover:bg-[var(--bg-active)] transition-colors border border-[var(--border)] text-[var(--text-secondary)]">
                {feat.icon}
              </div>
              <h3 className="text-xl font-semibold text-[var(--text-primary)] mt-2">
                {feat.title}
              </h3>
              <p className="text-[var(--text-secondary)] leading-relaxed">
                {feat.description}
              </p>
              <span className="text-xs text-[var(--text-muted)] mt-auto group-hover:text-[var(--text-secondary)] transition-colors">
                Click to learn more →
              </span>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Detail Modal Overlay */}
      <AnimatePresence>
        {expandedIdx !== null && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/30 backdrop-blur-sm z-50 flex items-center justify-center p-6"
            onClick={() => setExpandedIdx(null)}
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 10 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 10 }}
              transition={{ duration: 0.2 }}
              onClick={(e) => e.stopPropagation()}
              className="card max-w-lg w-full p-8 relative shadow-xl"
            >
              <button
                onClick={() => setExpandedIdx(null)}
                className="absolute top-4 right-4 p-1.5 rounded-lg hover:bg-[var(--bg-hover)] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
              >
                <X size={16} />
              </button>

              <div className="p-3 bg-[var(--bg-hover)] rounded-xl border border-[var(--border)] text-[var(--text-secondary)] w-fit mb-4">
                {features[expandedIdx].icon}
              </div>

              <h3 className="text-2xl font-bold text-[var(--text-primary)] mb-3">
                {features[expandedIdx].title}
              </h3>

              <p className="text-[var(--text-secondary)] leading-relaxed mb-4">
                {features[expandedIdx].description}
              </p>

              <div className="border-t border-dashed border-[var(--border-dotted)] pt-4">
                <p className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider mb-2">
                  Technical Details
                </p>
                <p className="text-sm text-[var(--text-secondary)] leading-relaxed">
                  {features[expandedIdx].detail}
                </p>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  );
}
