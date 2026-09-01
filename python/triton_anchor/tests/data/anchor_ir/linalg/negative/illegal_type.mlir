module {
  func.func @illegal_type(%pointer: !tt.ptr<f32>) {
    func.return
  }
}
