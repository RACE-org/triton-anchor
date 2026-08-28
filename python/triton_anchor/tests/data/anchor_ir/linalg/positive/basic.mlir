module {
  func.func @elementwise_add(
      %lhs: tensor<4xf32>,
      %rhs: tensor<4xf32>,
      %out: tensor<4xf32>) -> tensor<4xf32> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%lhs, %rhs : tensor<4xf32>, tensor<4xf32>)
      outs(%out : tensor<4xf32>) {
      ^bb0(%left: f32, %right: f32, %unused: f32):
        %sum = arith.addf %left, %right : f32
        linalg.yield %sum : f32
    } -> tensor<4xf32>
    func.return %result : tensor<4xf32>
  }
}
