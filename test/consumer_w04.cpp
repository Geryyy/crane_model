#include "crane_model/testing/mock_model.hpp"

int main()
{
  const auto model = crane_model::testing::MockModel::create(crane_model::Tool::Pzs100);
  if (!model.ok()) {
    return 1;
  }
  crane_model::ChamberPressure pressure;
  const auto result = model.value().transmission(
    crane_model::Q::Zero(), crane_model::DQA::Ones(), pressure);
  return result.ok() ? 0 : 1;
}
