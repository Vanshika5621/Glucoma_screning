import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ResultPanel } from "./ResultPanel";

describe("simple prediction result panel", () => {
  it("prompts for an image when the model is loaded but no result exists", () => {
    render(<ResultPanel result={null} modelAvailable />);

    expect(
      screen.getByText(/Upload an image to see its prediction and confidence/i),
    ).toBeInTheDocument();
  });

  it("explains that predictions are unavailable when no model is loaded", () => {
    render(<ResultPanel result={null} modelAvailable={false} />);

    expect(
      screen.getByText(/compatible trained research model is not loaded/i),
    ).toBeInTheDocument();
  });

  it("shows the Yes prediction with confidence and hides technical details", () => {
    render(
      <ResultPanel
        modelAvailable
        result={{
          prediction: "glaucoma",
          confidence: 0.8,
          glaucoma_probability: 0.8,
          threshold: 0.65,
          threshold_source: "validation-selected checkpoint threshold",
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

    expect(screen.getByRole("status")).toHaveTextContent("Yes");
    expect(screen.getByRole("status")).toHaveTextContent("Confidence: 80.0%");
    expect(screen.queryByText(/probability|threshold|Grad-CAM/i))
      .not.toBeInTheDocument();
  });

  it("shows No and the model confidence for a below-threshold score", () => {
    render(
      <ResultPanel
        modelAvailable
        result={{
          prediction: "normal",
          confidence: 0.529,
          glaucoma_probability: 0.471,
          threshold: 0.5,
          threshold_source: "validation-selected checkpoint threshold",
          disclaimer: "Research output only.",
          model_name: "resnet50",
          explanation_method: "Grad-CAM",
          explanation_note: "Approximate model explanation.",
          explanation_error: "",
          original_image: null,
          gradcam_heatmap: null,
          gradcam_overlay: null,
        }}
      />,
    );

    expect(screen.getByRole("status")).toHaveTextContent("No");
    expect(screen.getByRole("status")).toHaveTextContent("Confidence: 52.9%");
    expect(screen.queryByText(/Inconclusive|probability|threshold|Grad-CAM/i))
      .not.toBeInTheDocument();
  });
});
