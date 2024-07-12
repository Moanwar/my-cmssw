#ifndef RecoHGCal_TICL_TracksterInferenceByDNN_H__
#define RecoHGCal_TICL_TracksterInferenceByDNN_H__

#include "RecoHGCal/TICL/interface/TracksterInferenceAlgoBase.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

namespace ticl {
  class TracksterInferenceByDNN : public TracksterInferenceAlgoBase {
  public:
    explicit TracksterInferenceByDNN(const edm::ParameterSet& conf);
    void inputData(const std::vector<Trackster>& tracksters) override;
    void runInference(std::vector<Trackster>& tracksters) override;

  private:
    tensorflow::Session* session_;
  };
}  // namespace ticl

#endif
