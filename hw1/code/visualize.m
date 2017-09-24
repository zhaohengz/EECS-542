iter = 1:3000;
figure(1);
plot(iter, base_test, iter, leaky_test, iter, bn_test);
axis([1 3000 0 1]);
legend('Baseline', 'Leaky-ReLU', 'Batch Normalization');
xlabel('Iterations');
ylabel('Test Loss');


