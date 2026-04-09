"use client";

import {
  type AnalysisResult,
  analyzeVariantWithAPI,
  type ClinvarVariant,
  type GeneBounds,
  type GeneFromSearch,
} from "~/utils/genome-api";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Input } from "./ui/input";
import React, {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import {
  getClassificationColorClasses,
  getNucleotideColorClass,
} from "~/utils/coloring-utils";
import { Button } from "./ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { Zap } from "lucide-react";

export interface VariantAnalysisHandle {
  focusAlternativeInput: () => void;
}

export type AnalysisMode = "balanced" | "high_sensitivity" | "high_precision";

const MODE_LABELS: Record<AnalysisMode, string> = {
  balanced: "Balanced (Youden's J)",
  high_sensitivity: "High Sensitivity (clinical screening)",
  high_precision: "High Precision (research)",
};

const MODE_DESCRIPTIONS: Record<AnalysisMode, string> = {
  balanced: "Optimal trade-off between sensitivity and specificity.",
  high_sensitivity:
    "Maximises recall — fewer pathogenic variants missed. More false positives.",
  high_precision:
    "Minimises false positives — higher confidence predictions only.",
};

const FEATURE_LABELS: Record<string, string> = {
  evo2_delta_score: "Evo2 sequence disruption",
  gc_content: "GC content",
  position_fraction: "Positional context",
  phylop_score: "Conservation (PhyloP)",
};

interface VariantAnalysisProps {
  gene: GeneFromSearch;
  genomeId: string;
  chromosome: string;
  clinvarVariants: Array<ClinvarVariant>;
  referenceSequence: string | null;
  sequencePosition: number | null;
  geneBounds: GeneBounds | null;
  analysisMode: AnalysisMode;
  onAnalysisModeChange: (mode: AnalysisMode) => void;
}

const VariantAnalysis = forwardRef<VariantAnalysisHandle, VariantAnalysisProps>(
  (
    {
      gene,
      genomeId,
      chromosome,
      clinvarVariants = [],
      referenceSequence,
      sequencePosition,
      geneBounds,
      analysisMode,
      onAnalysisModeChange,
    }: VariantAnalysisProps,
    ref,
  ) => {
    const [variantPosition, setVariantPosition] = useState<string>(
      geneBounds?.min?.toString() || "",
    );
    const [variantReference, setVariantReference] = useState("");
    const [variantAlternative, setVariantAlternative] = useState("");
    const [variantResult, setVariantResult] = useState<AnalysisResult | null>(
      null,
    );
    const [isAnalyzingEvo2, setIsAnalyzingEvo2] = useState(false);
    const [isAnalyzingEnsemble, setIsAnalyzingEnsemble] = useState(false);
    const isAnalyzing = isAnalyzingEvo2 || isAnalyzingEnsemble;
    const [variantError, setVariantError] = useState<string | null>(null);
    const alternativeInputRef = useRef<HTMLInputElement>(null);

    useImperativeHandle(ref, () => ({
      focusAlternativeInput: () => {
        alternativeInputRef.current?.focus();
      },
    }));

    useEffect(() => {
      if (sequencePosition && referenceSequence) {
        setVariantPosition(String(sequencePosition));
        setVariantReference(referenceSequence);
      }
    }, [sequencePosition, referenceSequence]);

    const handlePositionChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      setVariantPosition(e.target.value);
      setVariantReference("");
    };

    const handleVariantSubmit = async (
      pos: string,
      alt: string,
      runEnsemble: boolean,
    ) => {
      const position = parseInt(pos);
      if (isNaN(position)) {
        setVariantError("Please enter a valid position number");
        return;
      }
      if (!/^[ATGC]$/.test(alt)) {
        setVariantError("Nucleotides must be A, C, G or T");
        return;
      }

      if (runEnsemble) setIsAnalyzingEnsemble(true);
      else setIsAnalyzingEvo2(true);
      setVariantError(null);

      try {
        const data = await analyzeVariantWithAPI({
          position,
          alternative: alt,
          genomeId,
          chromosome,
          mode: analysisMode,
          runEnsemble,
          geneStart: geneBounds?.min ?? undefined,
          geneEnd: geneBounds?.max ?? undefined,
        });
        setVariantResult(data);
      } catch (err) {
        console.error(err);
        setVariantError("Failed to analyze variant");
      } finally {
        setIsAnalyzingEvo2(false);
        setIsAnalyzingEnsemble(false);
      }
    };

    // Max absolute SHAP value — used to normalise the feature importance bars
    const maxShap = variantResult?.feature_importance
      ? Math.max(...Object.values(variantResult.feature_importance).map(Math.abs))
      : 1;

    return (
      <Card className="gap-0 border-none bg-white py-0 shadow-sm">
        <CardHeader className="pt-4 pb-2">
          <CardTitle className="text-sm font-normal text-[#3c4f3d]/70">
            Variant Analysis
          </CardTitle>
        </CardHeader>
        <CardContent className="pb-4">
          <p className="flex mb-4 text-xs text-[#3c4f3d]/80">
            Predict the impact of genetic variants using the Evo2 deep learning
            model.
          </p>

          {/* Input row */}
          <div className="flex flex-wrap items-start gap-4">
            <div>
              <label className="mb-1 block text-xs text-[#3c4f3d]/70">
                Position
              </label>
              <Input
                value={variantPosition}
                onChange={handlePositionChange}
                className="h-8 w-32 border-[#3c4f3d]/10 text-xs"
              />
            </div>

            <div>
              <label className="mb-1 block text-xs text-[#3c4f3d]/70">
                Alternative (variant)
              </label>
              <Input
                ref={alternativeInputRef}
                value={variantAlternative}
                onChange={(e) =>
                  setVariantAlternative(e.target.value.toUpperCase())
                }
                className="h-8 w-32 border-[#3c4f3d]/10 text-xs"
                placeholder="e.g., T"
                maxLength={1}
              />
            </div>

            {variantReference && (
              <div className="mb-2 flex items-center gap-2 text-xs text-[#3c4f3d]">
                <span>Substitution</span>
                <span
                  className={`font-medium ${getNucleotideColorClass(variantReference)}`}
                >
                  {variantReference}
                </span>
                <span>→</span>
                <span
                  className={`font-medium ${getNucleotideColorClass(variantAlternative)}`}
                >
                  {variantAlternative || "?"}
                </span>
              </div>
            )}

            {/* Mode selector */}
            <div>
              <label className="block text-xs text-[#3c4f3d]/70">
                Analysis mode
              </label>
              <Select
                value={analysisMode}
                onValueChange={(v) => onAnalysisModeChange(v as AnalysisMode)}
              >
                <SelectTrigger className="h-8 w-52 border-[#3c4f3d]/10 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {(
                    Object.entries(MODE_LABELS) as [AnalysisMode, string][]
                  ).map(([key, label]) => (
                    <SelectItem key={key} value={key} className="text-xs">
                      {label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="mt-1 max-w-xs text-[10px] text-[#3c4f3d]/50">
                {MODE_DESCRIPTIONS[analysisMode]}
              </p>
            </div>

            <div className="flex flex-col gap-1">
              <div className="flex gap-2">
                <Button
                  disabled={isAnalyzing || !variantPosition || !variantAlternative}
                  className="h-8 cursor-pointer bg-[#3c4f3d] text-xs text-white hover:bg-[#3c4f3d]/90"
                  onClick={() =>
                    handleVariantSubmit(
                      variantPosition.replaceAll(",", ""),
                      variantAlternative,
                      false,
                    )
                  }
                >
                  {isAnalyzingEvo2 ? (
                    <>
                      <span className="mr-2 inline-block h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent align-middle" />
                      Analyzing...
                    </>
                  ) : (
                    "Analyze with Evo2"
                  )}
                </Button>

                <Button
                  disabled={isAnalyzing || !variantPosition || !variantAlternative}
                  variant="outline"
                  className="h-8 cursor-pointer border-[#3c4f3d]/40 text-xs text-[#3c4f3d] hover:bg-[#3c4f3d]/10"
                  onClick={() =>
                    handleVariantSubmit(
                      variantPosition.replaceAll(",", ""),
                      variantAlternative,
                      true,
                    )
                  }
                >
                  {isAnalyzingEnsemble ? (
                    <>
                      <span className="mr-2 inline-block h-4 w-4 animate-spin rounded-full border-2 border-[#3c4f3d] border-t-transparent align-middle" />
                      Analyzing...
                    </>
                  ) : (
                    "Analyze with Ensemble"
                  )}
                </Button>
              </div>
              <p className="max-w-sm text-[10px] text-[#3c4f3d]/45">
                Ensemble model trained on BRCA1, BRCA2 &amp; TP53 variants.
                Most accurate for cancer-related genes.
              </p>
            </div>
          </div>

          {/* ClinVar match at this position */}
          {variantPosition &&
            clinvarVariants
              .filter(
                (variant) =>
                  variant?.variation_type
                    ?.toLowerCase()
                    .includes("single nucleotide") &&
                  parseInt(variant?.location?.replaceAll(",", "")) ===
                    parseInt(variantPosition.replaceAll(",", "")),
              )
              .map((matchedVariant) => {
                const refAltMatch = matchedVariant.title.match(/(\w)>(\w)/);
                const ref = refAltMatch?.[1] ?? null;
                const alt = refAltMatch?.[2] ?? null;
                if (!ref || !alt) return null;

                return (
                  <div
                    key={matchedVariant.clinvar_id}
                    className="mt-4 rounded-md border border-[#3c4f3d]/10 bg-[#e9eeea]/30 p-4"
                  >
                    <div className="mb-3 flex items-center justify-between">
                      <h4 className="text-sm font-medium text-[#3c4f3d]">
                        Known Variant Detected
                      </h4>
                      <span className="text-xs text-[#3c4f3d]/70">
                        Position: {matchedVariant.location}
                      </span>
                    </div>
                    <div className="grid gap-4 md:grid-cols-2">
                      <div>
                        <div className="mb-1 text-xs font-medium text-[#3c4f3d]/70">
                          Variant Details
                        </div>
                        <div className="text-sm">{matchedVariant.title}</div>
                        <div className="mt-2 text-sm">
                          {gene?.symbol} {variantPosition}{" "}
                          <span className="font-mono">
                            <span className={getNucleotideColorClass(ref)}>
                              {ref}
                            </span>
                            {">"}&nbsp;
                            <span className={getNucleotideColorClass(alt)}>
                              {alt}
                            </span>
                          </span>
                        </div>
                        <div className="mt-2 text-xs text-[#3c4f3d]/70">
                          ClinVar classification
                          <span
                            className={`ml-1 rounded-sm px-2 py-0.5 ${getClassificationColorClasses(matchedVariant.classification)}`}
                          >
                            {matchedVariant.classification || "Unknown"}
                          </span>
                        </div>
                      </div>
                      <div className="flex items-center justify-end">
                        <Button
                          disabled={isAnalyzing}
                          variant="outline"
                          size="sm"
                          className="h-7 cursor-pointer border-[#3c4f3d]/20 bg-[#e9eeea] text-xs text-[#3c4f3d] hover:bg-[#3c4f3d]/10"
                          onClick={() => {
                            setVariantAlternative(alt);
                            handleVariantSubmit(
                              variantPosition.replaceAll(",", ""),
                              alt,
                              false,
                            );
                          }}
                        >
                          {isAnalyzing ? (
                            <>
                              <span className="mr-1 inline-block h-3 w-3 animate-spin rounded-full border-2 border-white border-t-transparent align-middle" />
                              Analyzing...
                            </>
                          ) : (
                            <>
                              <Zap className="mr-1 inline-block h-3 w-3" />
                              Analyze this Variant
                            </>
                          )}
                        </Button>
                      </div>
                    </div>
                  </div>
                );
              })[0]}

          {/* Error */}
          {variantError && (
            <div className="mt-4 rounded-md bg-red-50 p-3 text-xs text-red-600">
              {variantError}
            </div>
          )}

          {/* Results */}
          {variantResult && (
            <div className="mt-6 space-y-4 rounded-md border border-[#3c4f3d]/10 bg-[#e9eeea]/30 p-4">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-medium text-[#3c4f3d]">
                  Analysis Result
                </h4>
              </div>

              {/* Top row: variant info + Evo2 score */}
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-3">
                  <div>
                    <div className="text-xs font-medium text-[#3c4f3d]/70">
                      Variant
                    </div>
                    <div className="text-sm">
                      {gene?.symbol} {variantResult.position}{" "}
                      <span className="font-mono">
                        {variantResult.reference}{">"}{variantResult.alternative}
                      </span>
                    </div>
                  </div>

                  <div>
                    <div className="text-xs font-medium text-[#3c4f3d]/70">
                      Evo2 delta likelihood score
                    </div>
                    <div className="text-sm font-mono">
                      {variantResult.delta_score.toFixed(6)}
                    </div>
                    <div className="text-xs text-[#3c4f3d]/60">
                      Negative score indicates loss of function
                    </div>
                  </div>

                  {variantResult.phylop_score !== null &&
                    variantResult.phylop_score !== undefined && (
                      <div>
                        <div className="text-xs font-medium text-[#3c4f3d]/70">
                          Conservation score{" "}
                          <span className="font-normal text-[#3c4f3d]/50">
                            (PhyloP100way)
                          </span>
                        </div>
                        <div className="text-sm font-mono">
                          {variantResult.phylop_score.toFixed(4)}
                        </div>
                        <div className="text-xs text-[#3c4f3d]/60">
                          {variantResult.phylop_score > 0
                            ? "Conserved across species — functionally constrained"
                            : "Rapidly evolving — less functional constraint"}
                        </div>
                      </div>
                    )}
                </div>

                <div className="space-y-3">
                  {/* Evo2 prediction */}
                  <div>
                    <div className="text-xs font-medium text-[#3c4f3d]/70">
                      Evo2 prediction
                      <span className="ml-2 text-[10px] font-normal text-[#3c4f3d]/40">
                        mode: {variantResult.operating_mode}
                      </span>
                    </div>
                    <div
                      className={`mt-1 inline-block rounded-lg px-3 py-1 text-xs ${getClassificationColorClasses(variantResult.prediction)}`}
                    >
                      {variantResult.prediction}
                    </div>
                  </div>

                  {/* Evo2 confidence bar */}
                  <div>
                    <div className="text-xs font-medium text-[#3c4f3d]/70">
                      Sequence-level confidence
                    </div>
                    <div className="mt-1 h-2 w-full rounded-full bg-[#e9eeea]">
                      <div
                        className={`h-2 rounded-full ${variantResult.prediction.includes("pathogenic") ? "bg-red-500" : "bg-green-500"}`}
                        style={{
                          width: `${Math.min(100, variantResult.classification_confidence * 100)}%`,
                        }}
                      />
                    </div>
                    <div className="mt-1 text-right text-xs text-[#3c4f3d]/60">
                      {Math.round(variantResult.classification_confidence * 100)}%
                    </div>
                  </div>

                  {/* XGBoost ensemble probability */}
                  {variantResult.xgboost_probability !== null &&
                    variantResult.xgboost_probability !== undefined && (
                      <div>
                        <div className="text-xs font-medium text-[#3c4f3d]/70">
                          Ensemble P(pathogenic)
                          <span className="ml-2 text-[10px] font-normal text-[#3c4f3d]/40">
                            Evo2 + conservation + position
                          </span>
                        </div>
                        <div className="mt-1 h-2 w-full rounded-full bg-[#e9eeea]">
                          <div
                            className={`h-2 rounded-full ${variantResult.xgboost_probability >= 0.5 ? "bg-red-500" : "bg-green-500"}`}
                            style={{
                              width: `${variantResult.xgboost_probability * 100}%`,
                            }}
                          />
                        </div>
                        <div className="mt-1 text-right text-xs text-[#3c4f3d]/60">
                          {Math.round(variantResult.xgboost_probability * 100)}%
                        </div>
                      </div>
                    )}
                </div>
              </div>

              {/* SHAP feature importance */}
              {variantResult.feature_importance && (
                <div>
                  <div className="mb-2 text-xs font-medium text-[#3c4f3d]/70">
                    Feature contributions{" "}
                    <span className="font-normal text-[#3c4f3d]/40">
                      (SHAP — positive pushes toward pathogenic)
                    </span>
                  </div>
                  <div className="space-y-2">
                    {Object.entries(variantResult.feature_importance)
                      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                      .map(([feature, shap]) => {
                        const pct = maxShap > 0 ? Math.abs(shap) / maxShap : 0;
                        const isPositive = shap >= 0;
                        return (
                          <div key={feature}>
                            <div className="mb-0.5 flex justify-between text-xs text-[#3c4f3d]/70">
                              <span>
                                {FEATURE_LABELS[feature] ?? feature}
                              </span>
                              <span
                                className={
                                  isPositive ? "text-red-500" : "text-green-600"
                                }
                              >
                                {isPositive ? "+" : ""}
                                {shap.toFixed(4)}
                              </span>
                            </div>
                            <div className="h-1.5 w-full rounded-full bg-[#e9eeea]">
                              <div
                                className={`h-1.5 rounded-full ${isPositive ? "bg-red-400" : "bg-green-500"}`}
                                style={{ width: `${pct * 100}%` }}
                              />
                            </div>
                          </div>
                        );
                      })}
                  </div>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    );
  },
);

VariantAnalysis.displayName = "VariantAnalysis";

export default VariantAnalysis;
