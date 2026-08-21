#include "crane_model/testing/mock_model.hpp"

int main()
{
  const auto model = crane_model::testing::MockModel::create(crane_model::Tool::Pzs100);
  if (!model.ok()) {
    return 1;
  }
  crane_model::Payload payload;
  payload.valid = true;
  payload.mass_kg = 1.0;
  const auto graph = model.value().symbolic_graph({}, payload);
  return graph.ok() ? 0 : 1;
}
