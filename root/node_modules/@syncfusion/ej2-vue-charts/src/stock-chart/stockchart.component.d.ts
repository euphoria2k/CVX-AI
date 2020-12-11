import { ComponentBase } from '@syncfusion/ej2-vue-base';
export declare const properties: string[];
export declare const modelProps: string[];
/**
 * Represents Vuejs chart Component
 * ```vue
 * <ejs-stockchart></ejs-stockchart>
 * ```
 */
export declare class StockChartComponent extends ComponentBase {
    ej2Instances: any;
    propKeys: string[];
    models: string[];
    hasChildDirective: boolean;
    protected hasInjectedModules: boolean;
    tagMapper: {
        [key: string]: Object;
    };
    tagNameMapper: Object;
    constructor();
    setProperties(prop: any, muteOnChange: boolean): void;
    trigger(eventName: string, eventProp: {
        [key: string]: Object;
    }, successHandler?: Function): void;
    render(createElement: any): any;
    chartModuleInjection(): void;
    findCurrentData(totalData: Object, xName: string): Object;
    rangeChanged(updatedStart: number, updatedEnd: number): void;
    renderPeriodSelector(): void;
    stockChartDataManagerSuccess(): void;
}
export declare const StockChartPlugin: {
    name: string;
    install(Vue: any): void;
};
