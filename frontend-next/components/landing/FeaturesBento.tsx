"use client";

import React from "react";
import { motion } from "framer-motion";
import { FileText, Cpu, Activity, Zap, Layers } from "lucide-react";

const features = [
  {
    title: "Multi-Format Documents",
    description: "Upload PDFs, Word docs, PowerPoints, and text. DocQuery's Unstructured.io pipeline intelligently extracts text, tables, and structure.",
    icon: <FileText size={24} />,
    className: "md:col-span-2",
  },
  {
    title: "Hybrid Retrieval",
    description: "Combines Pinecone vector search with BM25 keyword matching via Reciprocal Rank Fusion.",
    icon: <Layers size={24} />,
    className: "md:col-span-1",
  },
  {
    title: "RAGAS Evaluated",
    description: "Evaluated at 0.92 production-quality accuracy using the rigorous RAGAS framework.",
    icon: <Activity size={24} />,
    className: "md:col-span-1",
  },
  {
    title: "Celery Workers",
    description: "Horizontal scaling built-in. Heavy document chunking runs asynchronously without blocking the API.",
    icon: <Cpu size={24} />,
    className: "md:col-span-1",
  },
  {
    title: "Real-time SSE Streaming",
    description: 'Watch the AI "type" its answer in real-time, just like ChatGPT. Powered by Server-Sent Events for zero-latency perception.',
    icon: <Zap size={24} />,
    className: "md:col-span-1",
  },
];

export function FeaturesBento() {
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
            <div className="card h-full p-8 flex flex-col items-start gap-4 hover:-translate-y-1 transition-transform group">
              <div className="p-3 bg-[var(--bg-hover)] rounded-xl group-hover:bg-[var(--bg-active)] transition-colors border border-[var(--border)] text-[var(--text-secondary)]">
                {feat.icon}
              </div>
              <h3 className="text-xl font-semibold text-[var(--text-primary)] mt-2">
                {feat.title}
              </h3>
              <p className="text-[var(--text-secondary)] leading-relaxed">
                {feat.description}
              </p>
            </div>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
