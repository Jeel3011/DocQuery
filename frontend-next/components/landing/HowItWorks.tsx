"use client";

import React from "react";
import { motion } from "framer-motion";
import { Upload, MessageSquareText, BookOpenCheck } from "lucide-react";

const steps = [
  {
    title: "1. Upload",
    description: "Drop your documents in the sidebar. Celery workers instantly parse and chunk the text in the background.",
    icon: <Upload size={24} className="text-[var(--text-inverse)]" />,
  },
  {
    title: "2. Ask",
    description: "Ask a question. Our hybrid search pipeline uses embeddings and BM25 to find the most relevant chunks.",
    icon: <MessageSquareText size={24} className="text-[var(--text-inverse)]" />,
  },
  {
    title: "3. Cite",
    description: "GPT-4o-mini synthesizes an answer strictly grounded in the context, complete with inline source citations.",
    icon: <BookOpenCheck size={24} className="text-[var(--text-inverse)]" />,
  },
];

export function HowItWorks() {
  return (
    <section id="how-it-works" className="py-24 px-6 lg:px-12 max-w-7xl mx-auto relative z-10">
      <div className="text-center mb-20">
        <h2 className="text-3xl md:text-5xl font-bold text-[var(--text-primary)] mb-6 tracking-tight">
          How It Works
        </h2>
      </div>

      <div className="relative flex flex-col md:flex-row gap-12 md:gap-8 justify-between items-start">
        {/* Background connector line (Desktop) */}
        <div className="hidden md:block absolute top-[28px] left-[10%] right-[10%] h-[2px] border-t border-dashed border-[var(--border-dotted)]" />
        
        {/* Animated connector line (Desktop) */}
        <motion.div 
          initial={{ width: "0%" }}
          whileInView={{ width: "80%" }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 1.5, ease: "easeInOut" }}
          className="hidden md:block absolute top-[28px] left-[10%] h-[2px] bg-[var(--accent)] origin-left"
        />

        {steps.map((step, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.5, delay: i * 0.4 }}
            className="flex flex-col items-center text-center relative z-10 w-full md:w-1/3"
          >
            <motion.div 
              whileHover={{ scale: 1.1 }}
              className="w-14 h-14 rounded-2xl flex items-center justify-center mb-6 shadow-md bg-[var(--accent)]"
            >
              {step.icon}
            </motion.div>
            <h3 className="text-xl font-semibold text-[var(--text-primary)] mb-3">
              {step.title}
            </h3>
            <p className="text-[var(--text-secondary)] max-w-xs leading-relaxed">
              {step.description}
            </p>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
