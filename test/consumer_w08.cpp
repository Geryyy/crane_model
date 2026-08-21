#include "crane_model/testing/mock_model.hpp"

int main()
{
  const auto model = crane_model::testing::MockModel::create(crane_model::Tool::Pzs100);
  if (!model.ok()) {
    return 1;
  }
  const auto result = model.value().inverse_dynamics(
    crane_model::Q::Zero(), crane_model::DQ::Zero(), crane_model::DQ::Zero(),
    crane_model::Payload{});
  return result.status().code == crane_model::ErrorCode::InvalidPayload ? 0 : 1;
}
