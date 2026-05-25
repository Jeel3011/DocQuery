"use client";

import React from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowRight, Play } from "lucide-react";
import { HeroChatPreview } from "./HeroChatPreview";

export function HeroSection() {
  return (
    <section className="relative min-h-[90vh] flex items-center justify-center pt-20 pb-16 px-6 lg:px-12">
      <div className="max-w-7xl w-full grid grid-cols-1 lg:grid-cols-2 gap-12 lg:gap-8 items-center z-10">
        
        {/* Left Column: Copy & CTAs */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="flex flex-col items-center lg:items-start text-center lg:text-left gap-8"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-[var(--border)] bg-[var(--bg-surface)] text-[var(--text-secondary)] text-sm font-medium shadow-sm">
            <span className="flex h-2 w-2 rounded-full bg-[var(--accent)] animate-pulse"></span>
            DocQuery v2 is now live
          </div>

          <h1 className="text-5xl lg:text-7xl font-bold tracking-tight text-[var(--text-primary)] leading-[1.1]">
            Ask your documents <br />
            anything.
          </h1>

          <p className="text-lg lg:text-xl text-[var(--text-secondary)] max-w-xl leading-relaxed">
            DocQuery uses RAG — Retrieval-Augmented Generation — to give you precise, cited answers from your own documents. Upload, ask, understand.
          </p>

          <div className="flex flex-col sm:flex-row items-center gap-4 w-full sm:w-auto mt-4">
            <Link 
              href="/login" 
              className="btn-primary flex items-center justify-center gap-2 w-full sm:w-auto px-8 py-4 text-base"
            >
              Start for free
              <ArrowRight size={18} />
            </Link>
            
            <a 
              href="#how-it-works"
              className="btn-ghost flex items-center justify-center gap-2 w-full sm:w-auto px-8 py-4 text-base bg-[var(--bg-surface)] shadow-sm"
            >
              <Play size={18} />
              See how it works
            </a>
          </div>

          <p className="text-sm text-[var(--text-muted)] mt-2">
            No credit card required. Free tier available.
          </p>
        </motion.div>

        {/* Right Column: Animated Preview */}
        <motion.div 
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="flex justify-center lg:justify-end"
        >
          <div className="relative w-full max-w-lg">
            <HeroChatPreview />
          </div>
        </motion.div>

      </div>
    </section>
  );
}
