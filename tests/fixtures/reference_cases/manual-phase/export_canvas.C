// Extract observed Fourier coefficients through ROOT's public canvas interface.
#include <TCanvas.h>
#include <TFile.h>
#include <TH1.h>
#include <TKey.h>
#include <TList.h>
#include <TPad.h>
#include <fstream>
#include <stdexcept>

void collect_fourier(TObject *object, TH1 *&real, TH1 *&imag) {
  if (object->InheritsFrom(TPad::Class())) {
    for (auto child : *static_cast<TPad *>(object)->GetListOfPrimitives())
      collect_fourier(child, real, imag);
  } else if (object->InheritsFrom(TH1::Class())) {
    TString name = object->GetName();
    if (!name.Contains("_DataRunNo0_0_Fourier_")) return;
    if (name.EndsWith("_Re")) real = static_cast<TH1 *>(object);
    if (name.EndsWith("_Im")) imag = static_cast<TH1 *>(object);
  }
}

void export_canvas(const char *input, const char *output) {
  TFile file(input);
  TH1 *real = nullptr, *imag = nullptr;
  for (auto key : *file.GetListOfKeys())
    collect_fourier(static_cast<TKey *>(key)->ReadObj(), real, imag);
  if (!real || !imag || real->GetNbinsX() != imag->GetNbinsX())
    throw std::runtime_error("missing observed real/imaginary Fourier histograms");
  std::ofstream out(output);
  out.precision(17);
  for (int bin = 1; bin <= real->GetNbinsX(); ++bin) {
    if (real->GetBinCenter(bin) != imag->GetBinCenter(bin))
      throw std::runtime_error("inconsistent Fourier axes");
    out << real->GetBinCenter(bin) << "," << real->GetBinContent(bin)
        << "," << imag->GetBinContent(bin) << "\n";
  }
}
