#include <arrayfire_benchmark.h>

#include <arrayfire.h>
#if defined(ENABLE_ITK)
#include "itkImageFileReader.h"
#include "itkGPUGradientAnisotropicDiffusionImageFilter.h"
#endif

#include <array>
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
  in.eval();
  {
    //allocate output once to bypass alloc calls
    //when smoothing function is actually called
    array dummy(in.dims(), type);
    //run smoothing function to cache kernel compiled in first run
    array out =
        anisotropicDiffusion(in, TimeStep, Conductance, iterations);
    out.eval();
  }
  af::sync();
  for (auto _ : state) {
    array out = anisotropicDiffusion(in, TimeStep, Conductance, iterations);
    out.eval();
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
  //TODO(pradeep) enable other types later
  //vector<af_dtype> types = {f32, u32, u16, u8};
  vector<af_dtype> types = {f32};

  const vector<string> images = {
      ASSETS_DIR "/trees_ctm.jpg",
      ASSETS_DIR "/man.jpg",
  };
  const vector<std::array<string, 2>> asmImages = {
      {"4160x3120", ASSETS_DIR "/SizeVariation/01.jpg"},
      {"3648x2432", ASSETS_DIR "/SizeVariation/02.jpg"},
      {"2592x1944", ASSETS_DIR "/SizeVariation/03.jpg"},
      {"2048x1536", ASSETS_DIR "/SizeVariation/04.jpg"},
      {"1024x682" , ASSETS_DIR "/SizeVariation/05.jpg"},
      {"912x441"  , ASSETS_DIR "/SizeVariation/06.bmp"},
      {"553x411"  , ASSETS_DIR "/SizeVariation/07.bmp"},
      {"500x462"  , ASSETS_DIR "/SizeVariation/08.bmp"},
      {"413x339"  , ASSETS_DIR "/SizeVariation/09.bmp"},
      {"356x354"  , ASSETS_DIR "/SizeVariation/10.bmp"},
      {"277x145"  , ASSETS_DIR "/SizeVariation/11.bmp"},
      {"188x149"  , ASSETS_DIR "/SizeVariation/12.bmp"},
  };

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

  af::deviceGC();

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

  for(auto& img: asmImages) {
      auto name = "AF_" + img[0];
      af::benchmark::RegisterBenchmark(name.c_str(), types, afBase, img[1])
          ->RangeMultiplier(2)
          ->Range(32, 32)
          ->ArgName("Iterations")
          ->Unit(benchmark::kMillisecond);
  }

  benchmark::Initialize(&argc, argv);

  af::benchmark::AFReporter r;
  benchmark::RunSpecifiedBenchmarks(&r);
}
