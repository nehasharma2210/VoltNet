// Sample dataset for demonstration
const sampleData = {
    solar: {
        efficiency: 85,
        cost: 1200,
        co2Reduction: 2.5,
        reliability: 75
    },
    wind: {
        efficiency: 65,
        cost: 1500,
        co2Reduction: 3.2,
        reliability: 60
    },
    hybrid: {
        efficiency: 90,
        cost: 1800,
        co2Reduction: 3.0,
        reliability: 85
    }
};

// Function to get sample prediction
export function getSamplePrediction(source, demand) {
    const data = sampleData[source];
    const loadFactor = demand / 1000; // Scale factor
    
    return {
        efficiency: (data.efficiency + (Math.random() * 10 - 5)) / 100, // Convert to 0-1 range for consistency
        cost_savings: Math.round(data.cost * (0.5 + Math.random() * 0.5) * loadFactor),
        co2_reduction: data.co2Reduction * (0.8 + Math.random() * 0.4), // Return as number, not string
        reliability: data.reliability + (Math.random() * 10 - 5)
    };
}
