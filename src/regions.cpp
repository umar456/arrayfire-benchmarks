#include <arrayfire_benchmark.h>

#include <arrayfire.h>
#if defined(ENABLE_ITK)
#include "itkImageFileReader.h"
#include "itkConnectedComponentImageFilter.h"
#endif

#include <array>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

using std::string;
using std::vector;
using af::dim4;

constexpr unsigned int Dimension = 2;

static
void afBase(benchmark::State& state, af_dtype type, const string& image) {
  using af::regions;
  using af::loadImage;

  af::array in = loadImage(image.c_str(), false).as(type);
  {
    //allocate output once to bypass alloc calls
    //when smoothing function is actually called
    af::array dummy(in.dims(), type);
    //run smoothing function to cache kernel compiled in first run
    af::array out = regions(in, AF_CONNECTIVITY_8, u16);
  }
  af::sync();
  for (auto _ : state) {
    af::array out = regions(in, AF_CONNECTIVITY_8, u16);
    af::sync();
  }
  af::deviceGC();
}

#if defined(ENABLE_ITK)
static
void itkBase(benchmark::State& state, af_dtype type, const string& image) {
  using InputPixelType  = float;
  using InputImageType  = itk::Image< InputPixelType, Dimension >;
  using OutputPixelType = unsigned short;
  using OutputImageType = itk::Image< OutputPixelType, Dimension >;
  using ReaderType      = itk::ImageFileReader< InputImageType >;
  using RegionsFilter   =
      itk::ConnectedComponentImageFilter< InputImageType, OutputImageType >;

  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(image.c_str());
  reader->Update();

  RegionsFilter::Pointer filter = RegionsFilter::New();
  filter->SetInput(reader->GetOutput());

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
  const vector< std::array<string, 2> > benchInputs = {
      {"DesendingPlot", ASSETS_DIR "/ConnectedComponents/descending_1.png"},
      {"PolkaDots", ASSETS_DIR "/ConnectedComponents/dots_31.png"},
      {"EmptyImage", ASSETS_DIR "/ConnectedComponents/empty_0.png"},
      {"FullImage", ASSETS_DIR "/ConnectedComponents/full_1.png"},
      {"ImageFullofText", ASSETS_DIR "/ConnectedComponents/largetext_208.png"},
      {"TilesofText", ASSETS_DIR "/ConnectedComponents/texttiles_2140.png"},
      {"LargeVSpanningImage", ASSETS_DIR "/ConnectedComponents/v_1.png"},
  };

  vector<af_dtype> types = {b8};

  for (auto& benchIn: benchInputs) {
      af::benchmark::RegisterBenchmark(
              ("AF/"+benchIn[0]).c_str(), types, afBase, benchIn[1])
          ->Unit(benchmark::kMillisecond);
#if defined(ENABLE_ITK)
      af::benchmark::RegisterBenchmark(
              ("ITK/"+benchIn[0]).c_str(), types, itkBase, benchIn[1])
          ->Unit(benchmark::kMillisecond);
#endif
  }

  benchmark::Initialize(&argc, argv);

  af::benchmark::AFReporter r;
  benchmark::RunSpecifiedBenchmarks(&r);
}
