import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ResultPanel } from "./ResultPanel";

describe("Grad-CAM result panel", () => {
  it("shows original, heatmap, overlay, and the explanation caution", () => {
    render(
      <ResultPanel
        result={{
          prediction: "glaucoma",
          confidence: 0.8,
          disclaimer: "Research output only.",
          model_name: "resnet50",
          explanation_method: "Grad-CAM",
          explanation_note:
            "Approximate visualization; highlighted areas do not prove glaucoma.",
          explanation_error: "",
          original_image: "data:image/png;base64,original",
          gradcam_heatmap: "data:image/png;base64,heatmap",
          gradcam_overlay: "data:image/png;base64,overlay",
        }}
      />,
    );

    expect(screen.getByText("glaucoma")).toBeInTheDocument();
    expect(screen.getByText("Original image")).toBeInTheDocument();
    expect(screen.getByText("Grad-CAM heatmap")).toBeInTheDocument();
    expect(screen.getByText("Overlay")).toBeInTheDocument();
    expect(screen.getByLabelText("Grad-CAM heatmap legend")).toBeInTheDocument();
    expect(screen.getByText(/do not prove glaucoma/i)).toBeInTheDocument();
  });

  it("shows a clear message when visualization generation fails", () => {
    render(
      <ResultPanel
        result={{
          prediction: "normal",
          confidence: 0.7,
          disclaimer: "Research output only.",
          model_name: "resnet50",
          explanation_method: "Grad-CAM",
          explanation_note: "Approximate model explanation.",
          explanation_error:
            "Prediction completed, but its Grad-CAM visualization is unavailable.",
          original_image: null,
          gradcam_heatmap: null,
          gradcam_overlay: null,
        }}
      />,
    );

    expect(
      screen.getByText(/visualization is unavailable/i),
    ).toBeInTheDocument();
  });
});
