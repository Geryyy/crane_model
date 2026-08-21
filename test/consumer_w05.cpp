#include "crane_model/testing/mock_model.hpp"

int main()
{
  const auto model = crane_model::testing::MockModel::create(crane_model::Tool::Pzs100);
  if (!model.ok()) {
    return 1;
  }
  const auto result = model.value().collision_query(
    crane_model::Q::Zero(), crane_model::CollisionScene{});
  return result.ok() ? 0 : 1;
}
