#include <arrayfire_benchmark.h>

#include <arrayfire.h>
#if defined(ENABLE_ITK)
#include "itkImageFileReader.h"
#include "itkGPUGradientAnisotropicDiffusionImageFilter.h"
#endif

#include <cmath>
#include <string>
#include <vector>

using std::string;
using std::vector;
using af::array;
using af::dim4;

constexpr unsigned int Dimension = 2;
constexpr float TimeStep = 0.0105833f;
constexpr float Conductance = 0.35f;

static
void afBase(benchmark::State& state, af_dtype type, const string& image) {
  using af::anisotropicDiffusion;
  using af::loadImage;

  const unsigned iterations = unsigned(state.range(0));
  array in = loadImage(image.c_str(), false).as(type);
  {
    //allocate output once to bypass alloc calls
    //when smoothing function is actually called
    array dummy(in.dims(), type);
    //run smoothing function to cache kernel compiled in first run
    array out =
        anisotropicDiffusion(in, TimeStep, Conductance, iterations);
  }
  af::sync();
  for (auto _ : state) {
    array out = anisotropicDiffusion(in, TimeStep, Conductance, iterations);
    af::sync();
  }
  af::deviceGC();
}

#if defined(ENABLE_ITK)
static
void itkBase(benchmark::State& state, af_dtype type, const string& image) {
  using InputPixelType = float;
  using InputImageType = itk::GPUImage< InputPixelType, Dimension >;
  using ReaderType = itk::ImageFileReader< InputImageType >;
  using FilterType =
      itk::GPUGradientAnisotropicDiffusionImageFilter< InputImageType, InputImageType >;

  const unsigned iterations = unsigned(state.range(0));

  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(image.c_str());
  reader->Update();

  FilterType::Pointer filter = FilterType::New();
  filter->SetInput(reader->GetOutput());
  filter->SetNumberOfIterations(iterations);
  filter->SetTimeStep(TimeStep);
  filter->SetConductanceParameter(Conductance);

  for (auto _ : state) {
    try {
        reader->Modified();
        filter->UpdateLargestPossibleRegion();
    } catch (itk::ExceptionObject & error) {
        std::cerr << "Error: " << error << std::endl;
        throw;
    }
  }
}
#endif

int main(int argc, char** argv)
{
  const vector<string> images = {
      ASSETS_DIR "/trees_ctm.jpg",
      ASSETS_DIR "/man.jpg",
  };

  //TODO(pradeep) enable other types later
  //vector<af_dtype> types = {f32, u32, u16, u8};
  vector<af_dtype> types = {f32};

  af::benchmark::RegisterBenchmark("AF_1280x800", types, afBase, images[0])
      ->RangeMultiplier(2)
      ->Range(2, 1<<7)
      ->ArgName("Iterations")
      ->Unit(benchmark::kMillisecond);

  af::benchmark::RegisterBenchmark("AF_467x610", types, afBase, images[1])
      ->RangeMultiplier(2)
      ->Range(2, 1<<7)
      ->ArgName("Iterations")
      ->Unit(benchmark::kMillisecond);

#if defined(ENABLE_ITK)
  af::benchmark::RegisterBenchmark("ITK_1280x800", types, itkBase, images[0])
      ->RangeMultiplier(2)
      ->Range(2, 1<<7)
      ->ArgName("Iterations")
      ->Unit(benchmark::kMillisecond);

  af::benchmark::RegisterBenchmark("ITK_467x610", types, itkBase, images[1])
      ->RangeMultiplier(2)
      ->Range(2, 1<<7)
      ->ArgName("Iterations")
      ->Unit(benchmark::kMillisecond);
#endif

  benchmark::Initialize(&argc, argv);

  af::benchmark::AFReporter r;
  benchmark::RunSpecifiedBenchmarks(&r);
}
